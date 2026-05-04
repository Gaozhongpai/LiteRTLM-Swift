import Foundation
import CoreGraphics
import ImageIO
import os
import CLiteRTLM
#if canImport(Darwin)
import Darwin
#endif

/// Swift wrapper for Google's LiteRT-LM on-device inference engine.
///
/// Supports text generation (Session API) and multimodal inference — vision
/// and audio — (Conversation API) with `.litertlm` model files (e.g. Gemma 4 E2B).
///
/// Thread safety: all C API calls are serialized on an internal dispatch queue.
/// The class is `@unchecked Sendable` because OpaquePointers are only accessed
/// on that queue.
///
/// ## Quick Start
/// ```swift
/// let engine = LiteRTLMEngine(modelPath: modelURL)
/// try await engine.load()
///
/// // Text
/// let response = try await engine.generate(prompt: "Hello!", temperature: 0.7, maxTokens: 256)
///
/// // Vision
/// let caption = try await engine.vision(imageData: jpegData, prompt: "Describe this photo.")
///
/// // Audio
/// let transcript = try await engine.audio(audioData: wavData, prompt: "Transcribe this audio.")
/// ```
@Observable
public final class LiteRTLMEngine: @unchecked Sendable {

    // MARK: - Types

    public enum Status: Sendable, Equatable {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    /// A piece of input for a persistent session turn.
    ///
    /// Use with `sessionGenerateStreaming(inputs:)` to send mixed text and audio
    /// in a single user turn while reusing the session's KV cache.
    public enum SessionInput: Sendable {
        case text(String)
        case audio(Data)
        case preprocessedAudio(SessionPreprocessedAudio)
    }

    /// A compatibility wrapper for audio payloads created by
    /// `sessionPreprocessAudio(_:)`. Official LiteRT-LM currently accepts raw
    /// audio input directly through the session API, so this stores the original
    /// bytes instead of a native preprocessed handle.
    public final class SessionPreprocessedAudio: @unchecked Sendable {
        fileprivate let data: Data

        fileprivate init(data: Data) {
            self.data = data
        }
    }

    // MARK: - Properties

    public private(set) var status: Status = .notLoaded

    /// Whether the engine is ready for inference (text, vision, and audio).
    public var isReady: Bool { status == .ready }

    /// Whether the loaded backend supports `litert_lm_conversation_clone`.
    /// Probed once at load time. SessionBasic returns false; SessionAdvanced
    /// (when the model + execution manager support it) returns true.
    public private(set) var supportsConversationClone: Bool = false

    private let modelPath: URL
    private let backend: String

    private var engine: OpaquePointer?  // LiteRtLmEngine*
    private var didWarmTextDecode = false
    // The LiteRT-LM C runtime invokes streaming callbacks from its own worker
    // threads. Keep this coordinating queue at default QoS so its semaphore
    // wait does not create a user-initiated -> default priority inversion.
    private let inferenceQueue = DispatchQueue(label: "com.litertlm.inference", qos: .default)

    private static let log = Logger(subsystem: "LiteRTLMSwift", category: "Engine")
    private static let benchmarkLogsEnabled = false

    // MARK: - Init

    /// Create an engine instance.
    /// - Parameters:
    ///   - modelPath: Path to the `.litertlm` model file on disk.
    ///   - backend: Compute backend — `"cpu"` or `"gpu"` (GPU uses Metal on iOS).
    public init(modelPath: URL, backend: String = "cpu") {
        self.modelPath = modelPath
        self.backend = backend
    }

    deinit {
        let eng = engine
        let ses = chatSession
        let sesCfg = chatSessionConfig
        let conv = chatConversation
        let convCfg = chatConversationConfig
        let branches = Array(storedConversationBranches.values)
        let queue = inferenceQueue
        if eng != nil || ses != nil || conv != nil || !branches.isEmpty {
            queue.async {
                if let s = ses { litert_lm_session_delete(s) }
                if let c = sesCfg { litert_lm_session_config_delete(c) }
                if let c = conv { litert_lm_conversation_delete(c) }
                if let c = convCfg { litert_lm_conversation_config_delete(c) }
                for branch in branches { litert_lm_conversation_delete(branch) }
                if let e = eng { litert_lm_engine_delete(e) }
            }
        }
    }

    // MARK: - Lifecycle

    /// Load the `.litertlm` model. Call once, reuse for multiple inferences.
    /// Vision and audio encoders are embedded in the model file — no separate load step needed.
    @MainActor
    public func load() async throws {
        guard status != .ready && status != .loading else { return }

        status = .loading
        Self.log.info("Loading model: \(self.modelPath.lastPathComponent), backend: \(self.backend)")

        let path = modelPath.path
        let backendStr = self.backend
        let startTime = CFAbsoluteTimeGetCurrent()

        guard FileManager.default.fileExists(atPath: path) else {
            let msg = "Model file not found at \(path)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw LiteRTLMError.modelNotFound
        }

        do {
            let createdEngine = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<OpaquePointer, any Error>) in
                self.inferenceQueue.async {
                    do {
                        // Keep the native LiteRT runtime quiet so app-level
                        // branch and pipeline logs remain readable.
                        litert_lm_set_min_log_level(2)

                        guard let settings = litert_lm_engine_settings_create(
                            path, backendStr, backendStr, backendStr
                        ) else {
                            throw LiteRTLMError.engineCreationFailed("Failed to create engine settings")
                        }

                        litert_lm_engine_settings_set_max_num_tokens(settings, 4096)

                        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
                            .appendingPathComponent("litertlm_cache").path
                        try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
                        litert_lm_engine_settings_set_cache_dir(settings, cacheDir)

                        if Self.benchmarkLogsEnabled {
                            litert_lm_engine_settings_enable_benchmark(settings)
                        }

                        guard let createdEngine = litert_lm_engine_create(settings) else {
                            litert_lm_engine_settings_delete(settings)
                            throw LiteRTLMError.engineCreationFailed("litert_lm_engine_create returned NULL")
                        }
                        litert_lm_engine_settings_delete(settings)

                        continuation.resume(returning: createdEngine)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            inferenceQueue.sync {
                self.engine = createdEngine
                self.didWarmTextDecode = false
            }

            let cloneSupported = inferenceQueue.sync {
                Self.probeConversationCloneSupport(engine: createdEngine)
            }
            inferenceQueue.sync { self.supportsConversationClone = cloneSupported }
            Self.log.info("Conversation clone supported: \(cloneSupported)")

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            Self.log.info("Model loaded in \(String(format: "%.1f", elapsed))s")
            status = .ready
        } catch {
            let msg = "Load failed: \(error.localizedDescription)"
            Self.log.error("\(msg)")
            status = .error(msg)
            throw error
        }
    }

    /// Unload the model to free memory.
    @MainActor
    public func unload() {
        inferenceQueue.sync {
            if let s = chatSession {
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
            if let c = chatConversation {
                litert_lm_conversation_delete(c)
                chatConversation = nil
            }
            if let c = chatConversationConfig {
                litert_lm_conversation_config_delete(c)
                chatConversationConfig = nil
            }
            for branch in storedConversationBranches.values {
                litert_lm_conversation_delete(branch)
            }
            storedConversationBranches.removeAll()
            if let eng = engine { litert_lm_engine_delete(eng) }
            engine = nil
            didWarmTextDecode = false
        }
        supportsConversationClone = false
        status = .notLoaded
        Self.log.info("Model unloaded")
    }

    // MARK: - Text Generation (Session API)

    /// Generate text from a prompt. Creates a one-shot session per call.
    ///
    /// - Parameters:
    ///   - prompt: The input text. For Gemma 4, use `<|turn>user\n...<turn|>\n<|turn>model\n` format.
    ///   - temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative). Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text.
    public func generate(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws -> String {
        try ensureReady()
        return try await runSessionInference(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens),
            topK: Int32(topK), topP: topP
        )
    }

    /// Stream text generation token by token.
    ///
    /// Creates a one-shot session per call. For multi-turn conversations with
    /// KV cache reuse, use the persistent session API instead.
    ///
    /// - Parameters:
    ///   - prompt: The input text.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func generateStreaming(
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) -> AsyncThrowingStream<String, Error> {
        runSessionInferenceStreaming(
            prompt: prompt, temperature: temperature, maxTokens: Int32(maxTokens),
            topK: Int32(topK), topP: topP
        )
    }

    // MARK: - Vision (Conversation API)

    /// Run vision inference on a single image.
    ///
    /// Uses the Conversation API, which handles image decoding, resizing, and
    /// patchification internally. Input images are auto-converted to JPEG and
    /// resized to fit within `maxImageDimension`.
    ///
    /// - Parameters:
    ///   - imageData: Raw image bytes (JPEG, PNG, HEIC, etc.).
    ///   - prompt: Text prompt for the vision model (e.g., "Describe this photo.").
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func vision(
        imageData: Data,
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        let payload = try Self.prepareMultimodalPayload(
            audioData: [],
            audioFormat: .wav,
            imagesData: [imageData],
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return try await runConversationInference(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs,
            temperature: temperature,
            maxTokens: maxTokens,
            topK: topK,
            topP: topP
        )
    }

    /// Run vision inference on multiple images.
    ///
    /// - Parameters:
    ///   - imagesData: Array of raw image bytes.
    ///   - prompt: Text prompt about the images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func visionMultiImage(
        imagesData: [Data],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        topK: Int = 40,
        topP: Float = 0.95,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No images provided")
        }
        let payload = try Self.prepareMultimodalPayload(
            audioData: [],
            audioFormat: .wav,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return try await runConversationInference(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs,
            temperature: temperature,
            maxTokens: maxTokens,
            topK: topK,
            topP: topP
        )
    }

    // MARK: - Audio (Conversation API)

    /// Supported audio formats for the `audio()` and `multimodal()` methods.
    public enum AudioFormat: String, Sendable {
        case wav, flac, mp3
    }

    /// Run audio inference on a single audio file.
    ///
    /// Uses the Conversation API, which handles audio decoding and preprocessing
    /// (resample to 16 kHz, convert to mel spectrogram) internally.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV, FLAC, or MP3).
    ///   - prompt: Text prompt (e.g., "Transcribe this audio.", "Summarize what is being said.").
    ///   - format: Audio container format. Default `.wav`.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 512.
    /// - Returns: Generated text response.
    public func audio(
        audioData: Data,
        prompt: String,
        format: AudioFormat = .wav,
        temperature: Float = 0.7,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }
        let payload = try Self.prepareMultimodalPayload(
            audioData: [audioData],
            audioFormat: format,
            imagesData: [],
            prompt: prompt,
            maxImageDimension: 1024
        )
        return try await runConversationInference(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs,
            temperature: temperature,
            maxTokens: maxTokens,
            topK: topK,
            topP: topP
        )
    }

    /// Run multimodal inference combining audio, images, and text in a single query.
    ///
    /// Useful for tasks like "describe what's happening in this video" where you have
    /// both the audio track and keyframes, or "does this photo match what the speaker describes?".
    ///
    /// - Parameters:
    ///   - audioData: Array of raw audio bytes (WAV, FLAC, or MP3). Pass empty array to skip.
    ///   - imagesData: Array of raw image bytes (JPEG, PNG, HEIC). Pass empty array to skip.
    ///   - prompt: Text prompt about the audio and/or images.
    ///   - temperature: Sampling temperature. Default 0.7.
    ///   - maxTokens: Maximum tokens to generate. Default 1024.
    ///   - maxImageDimension: Resize image long edge to this value. Default 1024.
    /// - Returns: Generated text response.
    public func multimodal(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        temperature: Float = 0.7,
        maxTokens: Int = 1024,
        topK: Int = 40,
        topP: Float = 0.95,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty || !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio or image data provided")
        }
        let payload = try Self.prepareMultimodalPayload(
            audioData: audioData,
            audioFormat: audioFormat,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return try await runConversationInference(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs,
            temperature: temperature,
            maxTokens: maxTokens,
            topK: topK,
            topP: topP
        )
    }

    // MARK: - Persistent Session (KV Cache Reuse)
    //
    // LiteRT-LM's Session maintains a KV cache across multiple generate_content
    // calls. By keeping the session alive across turns, subsequent messages only
    // need to prefill NEW tokens instead of the entire conversation history.
    // This reduces TTFT from ~20s (full prefill) to ~1-2s (incremental).

    private var chatSession: OpaquePointer?
    private var chatSessionConfig: OpaquePointer?
    private var chatConversation: OpaquePointer?
    private var chatConversationConfig: OpaquePointer?
    private var storedConversationBranches: [String: OpaquePointer] = [:]

    /// Open a persistent session for multi-turn generation with KV cache reuse.
    ///
    /// Call once when a conversation begins. Subsequent calls to
    /// `sessionGenerateStreaming(input:)` reuse this session's KV cache.
    ///
    /// - Parameters:
    ///   - temperature: Sampling temperature. Default 0.3.
    ///   - maxTokens: Maximum tokens per generation. Default 512.
    public func openSession(
        temperature: Float = 0.3,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let c = chatConversation {
                        litert_lm_conversation_delete(c)
                        chatConversation = nil
                    }
                    if let c = chatConversationConfig {
                        litert_lm_conversation_config_delete(c)
                        chatConversationConfig = nil
                    }
                    if let s = chatSession {
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let (session, config) = try createSession(
                        engine: eng, temperature: temperature, maxTokens: Int32(maxTokens),
                        topK: Int32(topK), topP: topP
                    )
                    chatSession = session
                    chatSessionConfig = config
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent session, freeing KV cache memory.
    public func closeSession() {
        inferenceQueue.async { [self] in
            guard chatSession != nil else { return }
            if let s = chatSession {
                logSessionBenchmark(s)
                litert_lm_session_delete(s)
                chatSession = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
        }
    }

    /// Close the persistent session and wait until the engine queue finishes.
    public func closeSessionAndWait() async {
        await withCheckedContinuation { continuation in
            inferenceQueue.async { [self] in
                if let s = chatSession {
                    logSessionBenchmark(s)
                    litert_lm_session_delete(s)
                    chatSession = nil
                }
                if let c = chatSessionConfig {
                    litert_lm_session_config_delete(c)
                    chatSessionConfig = nil
                }
                continuation.resume()
            }
        }
    }

    // MARK: - Persistent Conversation (Multi-turn Multimodal)

    /// Open a persistent Conversation for multi-turn text and multimodal turns.
    ///
    /// Unlike the Session API, the Conversation API natively tracks history and
    /// supports text, image, and audio user turns in the same ongoing stateful
    /// interaction. `historyJSON`, when provided, should be a JSON array of prior
    /// messages using LiteRT-LM's Conversation message format.
    public func openConversation(
        systemPrompt: String? = nil,
        historyJSON: String? = nil,
        tools: [ToolDeclaration] = [],
        temperature: Float = 0.3,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    if let s = chatSession {
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }
                    if let c = chatConversation {
                        litert_lm_conversation_delete(c)
                        chatConversation = nil
                    }
                    if let c = chatConversationConfig {
                        litert_lm_conversation_config_delete(c)
                        chatConversationConfig = nil
                    }

                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let sessionConfig = try createSessionConfig(
                        temperature: temperature,
                        maxTokens: Int32(maxTokens),
                        topK: Int32(topK),
                        topP: topP
                    )
                    let toolsJSON = Self.buildToolsJSON(tools)
                    guard let conversationConfig = createConversationConfig(
                        engine: eng,
                        sessionConfig: sessionConfig,
                        systemPrompt: systemPrompt,
                        toolsJSON: toolsJSON,
                        historyJSON: historyJSON
                    ) else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create persistent conversation config")
                    }
                    guard let conversation = litert_lm_conversation_create(eng, conversationConfig) else {
                        litert_lm_conversation_config_delete(conversationConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create persistent conversation")
                    }

                    chatSessionConfig = sessionConfig
                    chatConversationConfig = conversationConfig
                    chatConversation = conversation
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Close the persistent Conversation and release its underlying state.
    public func closeConversation() {
        inferenceQueue.async { [self] in
            if let c = chatConversation {
                logConversationBenchmark(c)
                litert_lm_conversation_delete(c)
                chatConversation = nil
            }
            if let c = chatConversationConfig {
                litert_lm_conversation_config_delete(c)
                chatConversationConfig = nil
            }
            if let c = chatSessionConfig {
                litert_lm_session_config_delete(c)
                chatSessionConfig = nil
            }
        }
    }

    /// Close the persistent conversation and wait until the engine queue finishes.
    public func closeConversationAndWait() async {
        await withCheckedContinuation { continuation in
            inferenceQueue.async { [self] in
                if let c = chatConversation {
                    logConversationBenchmark(c)
                    litert_lm_conversation_delete(c)
                    chatConversation = nil
                }
                if let c = chatConversationConfig {
                    litert_lm_conversation_config_delete(c)
                    chatConversationConfig = nil
                }
                if let c = chatSessionConfig {
                    litert_lm_session_config_delete(c)
                    chatSessionConfig = nil
                }
                continuation.resume()
            }
        }
    }

    /// Create and store a named conversation branch.
    ///
    /// The branch is opened and prewarmed immediately using the provided
    /// system prompt, restored history, and tool declarations. You can later
    /// activate it into the engine's persistent conversation slot with
    /// `activateConversationBranch(_:)`.
    public func openConversationBranch(
        _ branchID: String,
        systemPrompt: String? = nil,
        historyJSON: String? = nil,
        tools: [ToolDeclaration] = [],
        temperature: Float = 0.3,
        maxTokens: Int = 512,
        topK: Int = 40,
        topP: Float = 0.95
    ) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    guard let eng = engine else { throw LiteRTLMError.modelNotLoaded }
                    let conversation = try createConversationHandle(
                        engine: eng,
                        systemPrompt: systemPrompt,
                        historyJSON: historyJSON,
                        tools: tools,
                        temperature: temperature,
                        maxTokens: maxTokens,
                        topK: topK,
                        topP: topP
                    )
                    deleteStoredConversationBranchLocked(branchID)
                    storedConversationBranches[branchID] = conversation
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Save a clone of the currently-open persistent conversation into a named
    /// branch. Requires a CLiteRTLM build that exports
    /// `litert_lm_conversation_clone`.
    public func saveConversationBranch(_ branchID: String) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    guard let conversation = chatConversation else {
                        throw LiteRTLMError.inferenceFailure("No persistent conversation open — call openConversation() first")
                    }
                    let clone = try cloneConversationHandle(conversation)
                    deleteStoredConversationBranchLocked(branchID)
                    storedConversationBranches[branchID] = clone
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Clone one stored conversation branch into another stored branch.
    ///
    /// This is the key primitive for "shared base prewarm then branch" flows:
    /// keep a base branch warm, clone it, then activate the clone and append
    /// page-specific turns.
    public func cloneConversationBranch(
        _ sourceBranchID: String,
        as targetBranchID: String
    ) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    guard let source = storedConversationBranches[sourceBranchID] else {
                        throw LiteRTLMError.inferenceFailure("Conversation branch '\(sourceBranchID)' was not found.")
                    }
                    let clone = try cloneConversationHandle(source)
                    deleteStoredConversationBranchLocked(targetBranchID)
                    storedConversationBranches[targetBranchID] = clone
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Activate a stored branch by cloning it into the engine's persistent
    /// conversation slot used by `conversationSend*` APIs.
    public func activateConversationBranch(_ branchID: String) async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            inferenceQueue.async { [self] in
                do {
                    guard let stored = storedConversationBranches[branchID] else {
                        throw LiteRTLMError.inferenceFailure("Conversation branch '\(branchID)' was not found.")
                    }
                    let clone = try cloneConversationHandle(stored)
                    if let c = chatConversation {
                        logConversationBenchmark(c)
                        litert_lm_conversation_delete(c)
                        chatConversation = nil
                    }
                    if let c = chatConversationConfig {
                        litert_lm_conversation_config_delete(c)
                        chatConversationConfig = nil
                    }
                    if let c = chatSessionConfig {
                        litert_lm_session_config_delete(c)
                        chatSessionConfig = nil
                    }
                    if let s = chatSession {
                        logSessionBenchmark(s)
                        litert_lm_session_delete(s)
                        chatSession = nil
                    }
                    chatConversation = clone
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Delete a stored conversation branch and free its warmed native state.
    public func deleteConversationBranch(_ branchID: String) {
        inferenceQueue.async { [self] in
            deleteStoredConversationBranchLocked(branchID)
        }
    }

    /// Returns true if a named conversation branch is currently stored.
    public func hasConversationBranch(_ branchID: String) async -> Bool {
        await withCheckedContinuation { continuation in
            inferenceQueue.async { [self] in
                continuation.resume(returning: storedConversationBranches[branchID] != nil)
            }
        }
    }

    public func conversationSendText(_ text: String) async throws -> String {
        try ensureReady()
        let messageJSON = Self.buildTextMessageJSON(text: text)
        return try await sendPersistentConversationMessage(messageJSON: messageJSON)
    }

    /// Send a user text message through a tool-enabled conversation and
    /// receive either natural-language text or a batch of tool calls.
    ///
    /// Prerequisites: the conversation was opened with `tools:` non-empty.
    /// Non-streaming: tool calls arrive in a single JSON object after the
    /// model finishes generating the turn.
    public func conversationSendTextWithTools(_ text: String) async throws -> ConversationTurn {
        try ensureReady()
        let messageJSON = Self.buildTextMessageJSON(text: text)
        let raw = try await sendPersistentConversationMessageRaw(messageJSON: messageJSON)
        return Self.parseConversationTurn(raw)
    }

    /// Send the results of a batch of tool calls back into the conversation
    /// and receive the model's next turn.
    ///
    /// Results are sent in positional order — the runtime does not correlate
    /// by id. The response may be another tool-call batch or final text.
    public func conversationSendToolResults(_ results: [ToolResult]) async throws -> ConversationTurn {
        try ensureReady()
        let messageJSON = Self.buildToolResultsMessageJSON(results)
        let raw = try await sendPersistentConversationMessageRaw(messageJSON: messageJSON)
        return Self.parseConversationTurn(raw)
    }

    /// Streaming counterpart of `conversationSendTextWithTools`.
    ///
    /// Yields `.text(chunk)` events as the model generates tokens and
    /// `.toolCalls(...)` events when the runtime has parsed complete tool
    /// calls. The stream ends when the current turn finishes — call
    /// `conversationSendToolResultsStreaming(_:)` for the next turn.
    ///
    /// Uses the same persistent Conversation as the non-streaming API so
    /// the KV cache survives across turns.
    public func conversationSendTextWithToolsStreaming(_ text: String) -> AsyncThrowingStream<ConversationTurn, Error> {
        let messageJSON = Self.buildTextMessageJSON(text: text)
        return streamPersistentConversationMessageWithTools(messageJSON: messageJSON)
    }

    /// Streaming counterpart of `conversationSendToolResults`.
    public func conversationSendToolResultsStreaming(_ results: [ToolResult]) -> AsyncThrowingStream<ConversationTurn, Error> {
        let messageJSON = Self.buildToolResultsMessageJSON(results)
        return streamPersistentConversationMessageWithTools(messageJSON: messageJSON)
    }

    /// Stream a text message through the persistent Conversation.
    ///
    /// The Conversation's KV cache holds all previous context. Each call appends
    /// a user turn and streams the model's response incrementally.
    ///
    /// - Parameter text: The user's text message.
    /// - Returns: An `AsyncThrowingStream` yielding response text chunks.
    public func conversationSendTextStreaming(_ text: String) -> AsyncThrowingStream<String, Error> {
        let messageJSON = Self.buildTextMessageJSON(text: text)
        return streamPersistentConversationMessage(messageJSON: messageJSON)
    }

    public func conversationSendAudio(
        audioData: Data,
        prompt: String,
        format: AudioFormat = .wav
    ) async throws -> String {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }

        let tempURL = Self.makeTempURL(extension: format.rawValue)
        try audioData.write(to: tempURL)
        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [tempURL.path],
            imagePaths: [],
            text: prompt
        )
        return try await sendPersistentConversationMessage(
            messageJSON: messageJSON,
            tempURLs: [tempURL]
        )
    }

    /// Stream an audio message through the persistent Conversation.
    ///
    /// The Conversation handles audio decoding and preprocessing internally.
    /// Its KV cache holds all previous context (text and audio turns).
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV, FLAC, or MP3).
    ///   - prompt: Text prompt accompanying the audio.
    ///   - format: Audio container format. Default `.wav`.
    /// - Returns: An `AsyncThrowingStream` yielding response text chunks.
    public func conversationSendAudioStreaming(
        audioData: Data,
        prompt: String,
        format: AudioFormat = .wav
    ) throws -> AsyncThrowingStream<String, Error> {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }

        let tempURL = Self.makeTempURL(extension: format.rawValue)
        try audioData.write(to: tempURL)
        let messageJSON = Self.buildMultimodalMessageJSON(
            audioPaths: [tempURL.path],
            imagePaths: [],
            text: prompt
        )
        return streamPersistentConversationMessage(
            messageJSON: messageJSON,
            tempURLs: [tempURL]
        )
    }

    /// Send a single image through the persistent Conversation.
    ///
    /// The Conversation's KV cache holds all previous context. The image is
    /// auto-converted to JPEG and resized before being attached to the user
    /// turn.
    public func conversationSendImage(
        imageData: Data,
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try await conversationSendImages(
            imagesData: [imageData],
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
    }

    /// Streaming counterpart of `conversationSendImage`.
    public func conversationSendImageStreaming(
        imageData: Data,
        prompt: String,
        maxImageDimension: Int = 1024
    ) throws -> AsyncThrowingStream<String, Error> {
        try ensureReady()
        return try conversationSendImagesStreaming(
            imagesData: [imageData],
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
    }

    /// Send one or more images through the persistent Conversation.
    ///
    /// The Conversation's KV cache holds all previous context (text, audio,
    /// and image turns). Input images are auto-converted to JPEG and resized.
    public func conversationSendImages(
        imagesData: [Data],
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        let payload = try Self.prepareMultimodalPayload(
            audioData: [],
            audioFormat: .wav,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return try await sendPersistentConversationMessage(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs
        )
    }

    /// Streaming counterpart of `conversationSendImages`.
    public func conversationSendImagesStreaming(
        imagesData: [Data],
        prompt: String,
        maxImageDimension: Int = 1024
    ) throws -> AsyncThrowingStream<String, Error> {
        try ensureReady()
        let payload = try Self.prepareMultimodalPayload(
            audioData: [],
            audioFormat: .wav,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return streamPersistentConversationMessage(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs
        )
    }

    /// Send a mixed audio + image turn through the persistent Conversation.
    ///
    /// Useful for stateful multimodal assistants that need to keep prior text,
    /// image, and audio context warm on the same Conversation.
    public func conversationSendMultimodal(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        maxImageDimension: Int = 1024
    ) async throws -> String {
        try ensureReady()
        let payload = try Self.prepareMultimodalPayload(
            audioData: audioData,
            audioFormat: audioFormat,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return try await sendPersistentConversationMessage(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs
        )
    }

    /// Streaming counterpart of `conversationSendMultimodal`.
    public func conversationSendMultimodalStreaming(
        audioData: [Data] = [],
        audioFormat: AudioFormat = .wav,
        imagesData: [Data] = [],
        prompt: String,
        maxImageDimension: Int = 1024
    ) throws -> AsyncThrowingStream<String, Error> {
        try ensureReady()
        let payload = try Self.prepareMultimodalPayload(
            audioData: audioData,
            audioFormat: audioFormat,
            imagesData: imagesData,
            prompt: prompt,
            maxImageDimension: maxImageDimension
        )
        return streamPersistentConversationMessage(
            messageJSON: payload.messageJSON,
            tempURLs: payload.tempURLs
        )
    }

    /// Stream text using the persistent session.
    ///
    /// `input` should be ONLY the new turn content — the session's KV cache
    /// already holds all previous context.
    ///
    /// - Parameter input: New input text for this turn.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func sessionGenerateStreaming(input: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let session = self.chatSession else {
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first"))
                    return
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let result = input.withCString { textPtr -> Int32 in
                    var inputData = LiteRtLmInputData(
                        type: kLiteRtLmInputDataTypeText,
                        data: UnsafeRawPointer(textPtr),
                        size: strlen(textPtr)
                    )
                    return litert_lm_session_generate_content_stream(
                        session, &inputData, 1,
                        { callbackData, chunk, isFinal, errorMsg in
                            guard let cbData = callbackData else { return }
                            let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                .takeUnretainedValue()

                            let errorMessage: String? = {
                                guard let errorMsg else { return nil }
                                let msg = String(cString: errorMsg)
                                return msg.isEmpty ? nil : msg
                            }()

                            if let chunk, errorMessage == nil {
                                let text = String(cString: chunk)
                                if !text.isEmpty { st.continuation.yield(text) }
                            }

                            if isFinal || errorMessage != nil {
                                if let error = errorMessage {
                                    st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                } else {
                                    st.continuation.finish()
                                }
                                let semaphore = st.doneSemaphore
                                Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                semaphore.signal()
                            }
                        },
                        statePtr
                    )
                }

                if result != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                    return
                }

                streamDone.wait()
                self.logSessionBenchmark(session)
            }
        }
    }

    /// Stream text from a persistent session turn with mixed text and audio inputs.
    ///
    /// The session's KV cache already holds all previous context. This method
    /// processes the inputs in order. Raw `.audio(Data)` is passed through to
    /// the underlying session API, while `.preprocessedAudio(...)` uses audio
    /// prepared by `sessionPreprocessAudio(_:)`.
    ///
    /// Example for a user turn containing audio + text prompt:
    /// ```swift
    /// let stream = engine.sessionGenerateStreaming(inputs: [
    ///     .text("<|turn>user\n"),
    ///     .preprocessedAudio(processedAudio),
    ///     .text("\nDescribe the audio.\n<turn|>\n<|turn>model\n")
    /// ])
    /// ```
    ///
    /// - Parameter inputs: Ordered array of text and audio inputs for this turn.
    /// - Returns: An `AsyncThrowingStream` yielding text chunks.
    public func sessionGenerateStreaming(inputs: [SessionInput]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let session = self.chatSession else {
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first"))
                    return
                }

                enum FlattenedInput {
                    case text(Data)
                    case rawAudio(Data)
                    case audioEnd
                    case preprocessedAudio(SessionPreprocessedAudio)
                }

                var inputEntries: [FlattenedInput] = []
                for item in inputs {
                    switch item {
                    case .text(let string):
                        inputEntries.append(.text(Data(string.utf8)))
                    case .audio(let data):
                        inputEntries.append(.rawAudio(data))
                        inputEntries.append(.audioEnd)
                    case .preprocessedAudio(let processed):
                        inputEntries.append(.preprocessedAudio(processed))
                        inputEntries.append(.audioEnd)
                    }
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                // Build C InputData structs. NSData gives stable .bytes pointers
                // that remain valid as long as the NSData objects are alive.
                var cInputs: [LiteRtLmInputData] = []
                cInputs.reserveCapacity(inputEntries.count)
                var pinnedBuffers: [NSData] = []
                var pinnedPreprocessedAudio: [SessionPreprocessedAudio] = []
                for entry in inputEntries {
                    switch entry {
                    case .text(let data):
                        var nullTerminated = data
                        nullTerminated.append(0)
                        let nsData = nullTerminated as NSData
                        pinnedBuffers.append(nsData)
                        cInputs.append(LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: nsData.bytes,
                            size: nsData.length - 1
                        ))
                    case .rawAudio(let data):
                        let nsData = data as NSData
                        pinnedBuffers.append(nsData)
                        cInputs.append(LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeAudio,
                            data: data.isEmpty ? nil : nsData.bytes,
                            size: nsData.length
                        ))
                    case .audioEnd:
                        cInputs.append(LiteRtLmInputData(type: kLiteRtLmInputDataTypeAudioEnd, data: nil, size: 0))
                    case .preprocessedAudio(let processed):
                        let nsData = processed.data as NSData
                        pinnedBuffers.append(nsData)
                        pinnedPreprocessedAudio.append(processed)
                        cInputs.append(LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeAudio,
                            data: processed.data.isEmpty ? nil : nsData.bytes,
                            size: nsData.length
                        ))
                    }
                }

                // Keep buffers and opaque preprocessed handles alive while the
                // C API reads input data.
                let callResult = withExtendedLifetime(pinnedBuffers) {
                    withExtendedLifetime(pinnedPreprocessedAudio) {
                        cInputs.withUnsafeMutableBufferPointer { bufferPtr -> Int32 in
                            guard let baseAddress = bufferPtr.baseAddress else { return -1 }
                            return litert_lm_session_generate_content_stream(
                                session, baseAddress, bufferPtr.count,
                                { callbackData, chunk, isFinal, errorMsg in
                                    guard let cbData = callbackData else { return }
                                    let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                        .takeUnretainedValue()

                                    let errorMessage: String? = {
                                        guard let errorMsg else { return nil }
                                        let msg = String(cString: errorMsg)
                                        return msg.isEmpty ? nil : msg
                                    }()

                                    if let chunk, errorMessage == nil {
                                        let text = String(cString: chunk)
                                        if !text.isEmpty { st.continuation.yield(text) }
                                    }

                                    if isFinal || errorMessage != nil {
                                        if let error = errorMessage {
                                            st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                        } else {
                                            st.continuation.finish()
                                        }
                                        let semaphore = st.doneSemaphore
                                        Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                        semaphore.signal()
                                    }
                                },
                                statePtr
                            )
                        }
                    }
                }

                if callResult != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start multimodal stream"))
                    return
                }

                streamDone.wait()
                self.logSessionBenchmark(session)
            }
        }
    }

    /// Returns an audio payload for mixed text+audio turns sent via
    /// `sessionGenerateStreaming(inputs:)`.
    public func sessionPreprocessAudio(_ audioData: Data) async throws -> SessionPreprocessedAudio {
        try ensureReady()
        guard !audioData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio data provided")
        }
        guard inferenceQueue.sync(execute: { self.chatSession != nil }) else {
            throw LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first")
        }
        return SessionPreprocessedAudio(data: audioData)
    }

    /// Prefill the persistent session without requesting a new model response.
    ///
    /// Use this to seed KV cache with system prompt and prior turns before the
    /// user sends the next message. The input should end on a closed turn marker
    /// and should not include a trailing `<|turn>model` opener.
    ///
    /// - Parameter input: Text to add to session context.
    public func sessionPrefill(input: String) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let session = self.chatSession else {
                        throw LiteRTLMError.inferenceFailure("No persistent session open — call openSession() first")
                    }

                    let result = input.withCString { textPtr -> Bool in
                        var inputData = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &inputData, 1) else {
                            return false
                        }
                        defer { litert_lm_responses_delete(responses) }
                        return true
                    }

                    guard result else {
                        throw LiteRTLMError.inferenceFailure("Failed to prefill session")
                    }

                    self.logSessionBenchmark(session)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Run a tiny throwaway decode so the first real streamed response does not
    /// have to initialize decode-side runtime state on the user path.
    public func warmUpTextDecodeIfNeeded() async throws {
        try ensureReady()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard !self.didWarmTextDecode else {
                        continuation.resume()
                        return
                    }
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng,
                        temperature: 0,
                        maxTokens: 1,
                        topK: 40,
                        topP: 0.95
                    )
                    defer {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let prompt = "<|turn>user\nHello<turn|>\n<|turn>model\n"
                    let didGenerate = prompt.withCString { textPtr -> Bool in
                        var input = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &input, 1) else {
                            return false
                        }
                        litert_lm_responses_delete(responses)
                        return true
                    }

                    guard didGenerate else {
                        throw LiteRTLMError.inferenceFailure("Text decode warmup returned no output")
                    }

                    self.didWarmTextDecode = true
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Private: Session-based Inference

    private func runSessionInference(
        prompt: String,
        temperature: Float,
        maxTokens: Int32,
        topK: Int32,
        topP: Float
    ) async throws -> String {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self] in
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens,
                        topK: topK, topP: topP
                    )
                    defer {
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    let output = prompt.withCString { textPtr -> String? in
                        var input = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        guard let responses = litert_lm_session_generate_content(session, &input, 1) else {
                            return nil
                        }
                        defer { litert_lm_responses_delete(responses) }
                        return self.extractResponseText(responses)
                    }

                    guard let result = output else {
                        throw LiteRTLMError.inferenceFailure("generate_content returned no output")
                    }

                    self.logSessionBenchmark(session)
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runSessionInferenceStreaming(
        prompt: String,
        temperature: Float,
        maxTokens: Int32,
        topK: Int32,
        topP: Float
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                do {
                    try self.ensureReady()
                    guard let eng = self.engine else {
                        continuation.finish(throwing: LiteRTLMError.modelNotLoaded)
                        return
                    }

                    let (session, sessionConfig) = try self.createSession(
                        engine: eng, temperature: temperature, maxTokens: maxTokens,
                        topK: topK, topP: topP
                    )

                    let streamDone = DispatchSemaphore(value: 0)
                    let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                    let statePtr = Unmanaged.passRetained(state).toOpaque()

                    let result = prompt.withCString { textPtr -> Int32 in
                        var input = LiteRtLmInputData(
                            type: kLiteRtLmInputDataTypeText,
                            data: UnsafeRawPointer(textPtr),
                            size: strlen(textPtr)
                        )
                        return litert_lm_session_generate_content_stream(
                            session, &input, 1,
                            { callbackData, chunk, isFinal, errorMsg in
                                guard let cbData = callbackData else { return }
                                let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                    .takeUnretainedValue()

                                let errorMessage: String? = {
                                    guard let errorMsg else { return nil }
                                    let msg = String(cString: errorMsg)
                                    return msg.isEmpty ? nil : msg
                                }()

                                if let chunk, errorMessage == nil {
                                    let text = String(cString: chunk)
                                    if !text.isEmpty { st.continuation.yield(text) }
                                }

                                if isFinal || errorMessage != nil {
                                    if let error = errorMessage {
                                        st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                    } else {
                                        st.continuation.finish()
                                    }
                                    let semaphore = st.doneSemaphore
                                    Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                    semaphore.signal()
                                }
                            },
                            statePtr
                        )
                    }

                    if result != 0 {
                        Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                        litert_lm_session_delete(session)
                        litert_lm_session_config_delete(sessionConfig)
                        continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start stream"))
                        return
                    }

                    streamDone.wait()
                    self.logSessionBenchmark(session)
                    litert_lm_session_delete(session)
                    litert_lm_session_config_delete(sessionConfig)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Private Helpers

    private func ensureReady() throws {
        guard status == .ready else { throw LiteRTLMError.modelNotLoaded }
    }

    private func createSession(
        engine eng: OpaquePointer,
        temperature: Float,
        maxTokens: Int32,
        topK: Int32,
        topP: Float
    ) throws -> (session: OpaquePointer, config: OpaquePointer) {
        let sessionConfig = try createSessionConfig(
            temperature: temperature,
            maxTokens: maxTokens,
            topK: topK,
            topP: topP
        )

        guard let session = litert_lm_engine_create_session(eng, sessionConfig) else {
            litert_lm_session_config_delete(sessionConfig)
            throw LiteRTLMError.inferenceFailure("Failed to create session")
        }

        return (session, sessionConfig)
    }

    private func createSessionConfig(
        temperature: Float,
        maxTokens: Int32,
        topK: Int32,
        topP: Float
    ) throws -> OpaquePointer {
        guard let sessionConfig = litert_lm_session_config_create() else {
            throw LiteRTLMError.inferenceFailure("Failed to create session config")
        }

        litert_lm_session_config_set_max_output_tokens(sessionConfig, maxTokens)
        var samplerParams = LiteRtLmSamplerParams(
            type: kLiteRtLmSamplerTypeTopP, top_k: topK, top_p: topP,
            temperature: temperature, seed: 0
        )
        litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)
        return sessionConfig
    }

    private func createConversationConfig(
        engine eng: OpaquePointer,
        sessionConfig: OpaquePointer,
        systemPrompt: String?,
        toolsJSON: String?,
        historyJSON: String?
    ) -> OpaquePointer? {
        let systemContent = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let toolsContent = toolsJSON?.trimmingCharacters(in: .whitespacesAndNewlines)
        let historyContent = historyJSON?.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let config = litert_lm_conversation_config_create() else {
            return nil
        }
        litert_lm_conversation_config_set_session_config(config, sessionConfig)
        litert_lm_conversation_config_set_enable_constrained_decoding(config, false)
        // Eagerly prefill system prompt + tools + history into the KV cache at
        // conversation-create time. Without this, prewarmProjectChat only
        // allocates structures; the model forward pass over the preface is
        // deferred to the first user turn. With this on, prewarm pays the
        // prefill cost up front so first-token latency drops to just the new
        // turn's prefill + decode.
        litert_lm_conversation_config_set_prefill_preface_on_init(config, true)

        Self.withOptionalCString(systemContent) { systemPtr in
            if let systemPtr {
                litert_lm_conversation_config_set_system_message(config, systemPtr)
            }
        }
        Self.withOptionalCString(toolsContent) { toolsPtr in
            if let toolsPtr {
                litert_lm_conversation_config_set_tools(config, toolsPtr)
            }
        }
        Self.withOptionalCString(historyContent) { historyPtr in
            if let historyPtr {
                litert_lm_conversation_config_set_messages(config, historyPtr)
            }
        }
        return config
    }

    private func createConversationHandle(
        engine eng: OpaquePointer,
        systemPrompt: String?,
        historyJSON: String?,
        tools: [ToolDeclaration],
        temperature: Float,
        maxTokens: Int,
        topK: Int,
        topP: Float
    ) throws -> OpaquePointer {
        let sessionConfig = try createSessionConfig(
            temperature: temperature,
            maxTokens: Int32(maxTokens),
            topK: Int32(topK),
            topP: topP
        )
        defer { litert_lm_session_config_delete(sessionConfig) }

        let toolsJSON = Self.buildToolsJSON(tools)
        guard let conversationConfig = createConversationConfig(
            engine: eng,
            sessionConfig: sessionConfig,
            systemPrompt: systemPrompt,
            toolsJSON: toolsJSON,
            historyJSON: historyJSON
        ) else {
            throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
        }
        defer { litert_lm_conversation_config_delete(conversationConfig) }

        guard let conversation = litert_lm_conversation_create(eng, conversationConfig) else {
            throw LiteRTLMError.inferenceFailure("Failed to create conversation")
        }
        return conversation
    }

    private func cloneConversationHandle(_ conversation: OpaquePointer) throws -> OpaquePointer {
        guard let cloned = litert_lm_conversation_clone(conversation) else {
            throw LiteRTLMError.inferenceFailure(
                "Conversation clone unsupported by current session backend (likely SessionBasic). Check engine.supportsConversationClone before calling clone APIs."
            )
        }
        return cloned
    }

    /// Probes whether the active engine backend implements `Session::Clone`.
    /// SessionBasic does not override Clone, so the call returns null.
    /// SessionAdvanced returns a real cloned conversation. Runs synchronously
    /// on the inference queue; safe to call once at load time.
    private static func probeConversationCloneSupport(engine eng: OpaquePointer) -> Bool {
        guard let sessionConfig = litert_lm_session_config_create() else { return false }
        defer { litert_lm_session_config_delete(sessionConfig) }

        guard let conversationConfig = litert_lm_conversation_config_create() else { return false }
        litert_lm_conversation_config_set_session_config(conversationConfig, sessionConfig)
        defer { litert_lm_conversation_config_delete(conversationConfig) }

        guard let conversation = litert_lm_conversation_create(eng, conversationConfig) else { return false }
        defer { litert_lm_conversation_delete(conversation) }

        guard let cloned = litert_lm_conversation_clone(conversation) else { return false }
        litert_lm_conversation_delete(cloned)
        return true
    }

    private func deleteStoredConversationBranchLocked(_ branchID: String) {
        guard let conversation = storedConversationBranches.removeValue(forKey: branchID) else { return }
        litert_lm_conversation_delete(conversation)
    }

    private nonisolated static func withOptionalCString<T>(
        _ value: String?,
        _ body: (UnsafePointer<CChar>?) -> T
    ) -> T {
        if let value {
            return value.withCString { body($0) }
        }
        return body(nil)
    }

    private func extractResponseText(_ responses: OpaquePointer) -> String? {
        let numCandidates = litert_lm_responses_get_num_candidates(responses)
        guard numCandidates > 0,
              let resultPtr = litert_lm_responses_get_response_text_at(responses, 0) else {
            return nil
        }
        return String(cString: resultPtr)
    }

    private func logSessionBenchmark(_ session: OpaquePointer) {
        _ = session
    }

    private func logConversationBenchmark(_ conversation: OpaquePointer) {
        _ = conversation
    }

    // MARK: - Private: Conversation-based Inference (Vision / Audio / Multimodal)

    /// Shared helper for all Conversation API calls (vision, audio, multimodal).
    /// Handles session/conversation lifecycle and temp file cleanup.
    private func runConversationInference(
        messageJSON: String,
        tempURLs: [URL],
        temperature: Float,
        maxTokens: Int,
        topK: Int,
        topP: Float
    ) async throws -> String {
        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer {
                    for url in urlsToCleanup {
                        try? FileManager.default.removeItem(at: url)
                    }
                }
                do {
                    guard let eng = self.engine else { throw LiteRTLMError.modelNotLoaded }

                    guard let sessionConfig = litert_lm_session_config_create() else {
                        throw LiteRTLMError.inferenceFailure("Failed to create session config")
                    }
                    litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxTokens))
                    var samplerParams = LiteRtLmSamplerParams(
                        type: kLiteRtLmSamplerTypeTopP, top_k: Int32(topK), top_p: topP,
                        temperature: temperature, seed: 0
                    )
                    litert_lm_session_config_set_sampler_params(sessionConfig, &samplerParams)

                    guard let convConfig = litert_lm_conversation_config_create() else {
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation config")
                    }
                    litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)
                    litert_lm_conversation_config_set_enable_constrained_decoding(convConfig, false)

                    guard let conversation = litert_lm_conversation_create(eng, convConfig) else {
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                        throw LiteRTLMError.inferenceFailure("Failed to create conversation")
                    }
                    defer {
                        litert_lm_conversation_delete(conversation)
                        litert_lm_conversation_config_delete(convConfig)
                        litert_lm_session_config_delete(sessionConfig)
                    }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Response string is NULL")
                    }

                    let result = Self.extractTextFromConversationResponse(String(cString: responsePtr))
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func sendPersistentConversationMessage(
        messageJSON: String,
        tempURLs: [URL] = []
    ) async throws -> String {
        let raw = try await sendPersistentConversationMessageRaw(
            messageJSON: messageJSON,
            tempURLs: tempURLs
        )
        return Self.extractTextFromConversationResponse(raw)
    }

    /// Like `sendPersistentConversationMessage` but returns the raw response
    /// JSON string instead of extracting text. Callers that need to see
    /// `tool_calls` use this path and parse themselves.
    private func sendPersistentConversationMessageRaw(
        messageJSON: String,
        tempURLs: [URL] = []
    ) async throws -> String {
        let urlsToCleanup = tempURLs
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
            self.inferenceQueue.async { [self, urlsToCleanup] in
                defer { Self.cleanupTempFiles(urlsToCleanup) }
                do {
                    guard let conversation = self.chatConversation else {
                        throw LiteRTLMError.inferenceFailure("No persistent conversation open — call openConversation() first")
                    }

                    guard let response = messageJSON.withCString({ msgPtr in
                        litert_lm_conversation_send_message(conversation, msgPtr, nil)
                    }) else {
                        throw LiteRTLMError.inferenceFailure("Persistent conversation returned no response")
                    }
                    defer { litert_lm_json_response_delete(response) }

                    guard let responsePtr = litert_lm_json_response_get_string(response) else {
                        throw LiteRTLMError.inferenceFailure("Persistent conversation response string is NULL")
                    }

                    let raw = String(cString: responsePtr)
                    self.logConversationBenchmark(conversation)
                    continuation.resume(returning: raw)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Streaming variant that yields `ConversationTurn` events — text chunks
    /// and tool-call batches — on the persistent Conversation's KV cache.
    private func streamPersistentConversationMessageWithTools(
        messageJSON: String
    ) -> AsyncThrowingStream<ConversationTurn, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                guard let conversation = self.chatConversation else {
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent conversation open — call openConversation() first"))
                    return
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = ToolStreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let callResult = messageJSON.withCString { msgPtr -> Int32 in
                    litert_lm_conversation_send_message_stream(
                        conversation, msgPtr, nil,
                        { callbackData, chunk, isFinal, errorMsg in
                            guard let cbData = callbackData else { return }
                            let st = Unmanaged<ToolStreamCallbackState>.fromOpaque(cbData)
                                .takeUnretainedValue()

                            let errorMessage: String? = {
                                guard let errorMsg else { return nil }
                                let msg = String(cString: errorMsg)
                                return msg.isEmpty ? nil : msg
                            }()

                            if let chunk, errorMessage == nil {
                                let raw = String(cString: chunk)
                                let event = LiteRTLMEngine.parseConversationTurn(raw)
                                switch event {
                                case .text(let t) where !t.isEmpty:
                                    st.continuation.yield(.text(t))
                                case .toolCalls(let calls) where !calls.isEmpty:
                                    st.continuation.yield(.toolCalls(calls))
                                default:
                                    break
                                }
                            }

                            if isFinal || errorMessage != nil {
                                if let error = errorMessage {
                                    st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                } else {
                                    st.continuation.finish()
                                }
                                let semaphore = st.doneSemaphore
                                Unmanaged<ToolStreamCallbackState>.fromOpaque(cbData).release()
                                semaphore.signal()
                            }
                        },
                        statePtr
                    )
                }

                if callResult != 0 {
                    Unmanaged<ToolStreamCallbackState>.fromOpaque(statePtr).release()
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start conversation stream"))
                    return
                }

                streamDone.wait()
                self.logConversationBenchmark(conversation)
            }
        }
    }

    /// Streaming counterpart of `sendPersistentConversationMessage`.
    ///
    /// Uses `litert_lm_conversation_send_message_stream` to yield tokens
    /// incrementally instead of blocking for the full response.
    private func streamPersistentConversationMessage(
        messageJSON: String,
        tempURLs: [URL] = []
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            self.inferenceQueue.async { [self] in
                let urlsToCleanup = tempURLs
                guard let conversation = self.chatConversation else {
                    Self.cleanupTempFiles(urlsToCleanup)
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("No persistent conversation open — call openConversation() first"))
                    return
                }

                let streamDone = DispatchSemaphore(value: 0)
                let state = StreamCallbackState(continuation: continuation, doneSemaphore: streamDone)
                let statePtr = Unmanaged.passRetained(state).toOpaque()

                let callResult = messageJSON.withCString { msgPtr -> Int32 in
                    litert_lm_conversation_send_message_stream(
                        conversation, msgPtr, nil,
                        { callbackData, chunk, isFinal, errorMsg in
                            guard let cbData = callbackData else { return }
                            let st = Unmanaged<StreamCallbackState>.fromOpaque(cbData)
                                .takeUnretainedValue()

                            let errorMessage: String? = {
                                guard let errorMsg else { return nil }
                                let msg = String(cString: errorMsg)
                                return msg.isEmpty ? nil : msg
                            }()

                            if let chunk, errorMessage == nil {
                                let raw = String(cString: chunk)
                                let text = LiteRTLMEngine.extractTextFromConversationResponse(raw)
                                if !text.isEmpty { st.continuation.yield(text) }
                            }

                            if isFinal || errorMessage != nil {
                                if let error = errorMessage {
                                    st.continuation.finish(throwing: LiteRTLMError.inferenceFailure(error))
                                } else {
                                    st.continuation.finish()
                                }
                                let semaphore = st.doneSemaphore
                                Unmanaged<StreamCallbackState>.fromOpaque(cbData).release()
                                semaphore.signal()
                            }
                        },
                        statePtr
                    )
                }

                if callResult != 0 {
                    Unmanaged<StreamCallbackState>.fromOpaque(statePtr).release()
                    Self.cleanupTempFiles(urlsToCleanup)
                    continuation.finish(throwing: LiteRTLMError.inferenceFailure("Failed to start conversation stream"))
                    return
                }

                streamDone.wait()
                Self.cleanupTempFiles(urlsToCleanup)
                self.logConversationBenchmark(conversation)
            }
        }
    }

    // MARK: - Media Helpers

    /// Create a uniquely-named temp file URL.
    nonisolated static func makeTempURL(extension ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + "." + ext)
    }

    /// Remove temp files, ignoring errors (best-effort cleanup).
    nonisolated static func cleanupTempFiles(_ urls: [URL]) {
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Convert any image format to JPEG and resize for vision inference.
    nonisolated static func prepareImageForVision(_ data: Data, maxDimension: Int = 1024) -> Data? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }

        let width = cgImage.width
        let height = cgImage.height

        let maxDim = maxDimension
        let scale: Double
        if width > height {
            scale = width > maxDim ? Double(maxDim) / Double(width) : 1.0
        } else {
            scale = height > maxDim ? Double(maxDim) / Double(height) : 1.0
        }

        let targetWidth = Int(Double(width) * scale)
        let targetHeight = Int(Double(height) * scale)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: targetWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        guard let resizedImage = context.makeImage() else { return nil }

        let mutableData = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            mutableData, "public.jpeg" as CFString, 1, nil
        ) else { return nil }

        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.85]
        CGImageDestinationAddImage(destination, resizedImage, options as CFDictionary)

        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }

    nonisolated static func prepareMultimodalPayload(
        audioData: [Data],
        audioFormat: AudioFormat,
        imagesData: [Data],
        prompt: String,
        maxImageDimension: Int
    ) throws -> (messageJSON: String, tempURLs: [URL]) {
        guard !audioData.isEmpty || !imagesData.isEmpty else {
            throw LiteRTLMError.inferenceFailure("No audio or image data provided")
        }

        var tempURLs: [URL] = []
        var audioPaths: [String] = []
        var imagePaths: [String] = []

        do {
            for (i, data) in audioData.enumerated() {
                guard !data.isEmpty else {
                    throw LiteRTLMError.inferenceFailure("Audio data \(i + 1) is empty")
                }
                let url = Self.makeTempURL(extension: audioFormat.rawValue)
                try data.write(to: url)
                tempURLs.append(url)
                audioPaths.append(url.path)
            }

            for (i, data) in imagesData.enumerated() {
                guard let jpegData = Self.prepareImageForVision(data, maxDimension: maxImageDimension) else {
                    throw LiteRTLMError.inferenceFailure("Failed to convert image \(i + 1) to JPEG")
                }
                let url = Self.makeTempURL(extension: "jpg")
                try jpegData.write(to: url)
                tempURLs.append(url)
                imagePaths.append(url.path)
            }
        } catch {
            Self.cleanupTempFiles(tempURLs)
            throw error
        }

        return (
            messageJSON: Self.buildMultimodalMessageJSON(
                audioPaths: audioPaths,
                imagePaths: imagePaths,
                text: prompt
            ),
            tempURLs: tempURLs
        )
    }

    /// Build a Conversation API JSON message with any combination of audio, images, and text.
    nonisolated static func buildMultimodalMessageJSON(
        audioPaths: [String],
        imagePaths: [String],
        text: String
    ) -> String {
        var contentItems: [[String: Any]] = []
        for path in audioPaths {
            contentItems.append(["type": "audio", "path": path])
        }
        for path in imagePaths {
            contentItems.append(["type": "image", "path": path])
        }
        contentItems.append(["type": "text", "text": text])
        let message: [String: Any] = ["role": "user", "content": contentItems]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: message),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            // Fallback: text-only, properly escaped via JSONSerialization
            let fallback: [String: Any] = ["role": "user", "content": [["type": "text", "text": text]]]
            let fallbackData = (try? JSONSerialization.data(withJSONObject: fallback)) ?? Data()
            return String(data: fallbackData, encoding: .utf8) ?? "{}"
        }
        return jsonString
    }

    nonisolated static func buildTextMessageJSON(text: String) -> String {
        #"{"role":"user","content":"# + jsonStringLiteral(text) + "}"
    }

    /// Serialize tool declarations into the JSON array format the C API expects
    /// for `tools_json`. Each entry is wrapped as
    /// `{"type": "function", "function": {name, description, parameters}}` —
    /// the shape Gemma 4's bundled Jinja chat template iterates over.
    /// Returns nil for an empty list.
    nonisolated static func buildToolsJSON(_ tools: [ToolDeclaration]) -> String? {
        guard !tools.isEmpty else { return nil }
        var items: [[String: Any]] = []
        for decl in tools {
            guard let paramsData = decl.parametersJSON.data(using: .utf8),
                  let paramsObj = try? JSONSerialization.jsonObject(with: paramsData) else {
                continue
            }
            items.append([
                "type": "function",
                "function": [
                    "name": decl.name,
                    "description": decl.description,
                    "parameters": paramsObj
                ]
            ])
        }
        guard !items.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: items),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return json
    }

    /// Build the message JSON for sending tool results back into a conversation.
    /// Single result becomes a single message; multiple results become an array
    /// of messages — both accepted by the runtime.
    nonisolated static func buildToolResultsMessageJSON(_ results: [ToolResult]) -> String {
        let messages: [[String: Any]] = results.compactMap { result in
            guard let data = result.contentJSON.data(using: .utf8),
                  let content = try? JSONSerialization.jsonObject(with: data) else {
                return nil
            }
            return ["role": "tool", "content": content]
        }
        let payload: Any = messages.count == 1 ? messages[0] : messages
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8) else {
            return "[]"
        }
        return json
    }

    /// Parse a raw conversation response into either text or tool calls.
    /// Falls through to `.text(raw)` for any shape we don't recognize.
    nonisolated static func parseConversationTurn(_ json: String) -> ConversationTurn {
        if !json.contains(#""tool_calls""#),
           let text = fastExtractTextFromConversationResponse(json) {
            return .text(text)
        }

        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .text(json.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        if let calls = obj["tool_calls"] as? [[String: Any]], !calls.isEmpty {
            var parsed: [ToolCall] = []
            for entry in calls {
                guard let function = entry["function"] as? [String: Any],
                      let name = function["name"] as? String else { continue }
                let argumentsJSON: String
                if let argsObj = function["arguments"] {
                    let cleaned = stripGemmaStringDelimiters(in: argsObj)
                    if let argsData = try? JSONSerialization.data(withJSONObject: cleaned),
                       let argsString = String(data: argsData, encoding: .utf8) {
                        argumentsJSON = argsString
                    } else if let argsString = argsObj as? String {
                        argumentsJSON = cleanGemmaString(argsString)
                    } else {
                        argumentsJSON = "{}"
                    }
                } else {
                    argumentsJSON = "{}"
                }
                parsed.append(ToolCall(name: name, argumentsJSON: argumentsJSON))
            }
            if !parsed.isEmpty { return .toolCalls(parsed) }
        }

        return .text(extractTextFromConversationResponse(json))
    }

    /// Recursively strip Gemma's `<|"|>` string-quote token from every
    /// string leaf in a JSON value. The runtime's tool-call parser leaves
    /// these delimiters in string arguments on some models.
    private nonisolated static func stripGemmaStringDelimiters(in value: Any) -> Any {
        if let s = value as? String {
            return cleanGemmaString(s)
        }
        if let arr = value as? [Any] {
            return arr.map { stripGemmaStringDelimiters(in: $0) }
        }
        if let dict = value as? [String: Any] {
            var out: [String: Any] = [:]
            for (k, v) in dict {
                out[k] = stripGemmaStringDelimiters(in: v)
            }
            return out
        }
        return value
    }

    private nonisolated static func cleanGemmaString(_ s: String) -> String {
        s.replacingOccurrences(of: "<|\"|>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func extractTextFromConversationResponse(_ json: String) -> String {
        if let text = fastExtractTextFromConversationResponse(json) {
            return text
        }

        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        if let content = obj["content"] as? [[String: Any]] {
            let texts = content.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let candidates = obj["candidates"] as? [[String: Any]],
           let first = candidates.first,
           let content = first["content"] as? [String: Any],
           let parts = content["parts"] as? [[String: Any]] {
            let texts = parts.compactMap { $0["text"] as? String }
            if !texts.isEmpty { return texts.joined(separator: " ") }
        }

        if let text = obj["text"] as? String { return text }

        return json.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func fastExtractTextFromConversationResponse(_ json: String) -> String? {
        if let text = fastExtractJSONStringField("text", in: json) {
            return text
        }
        if let text = fastExtractJSONStringField("content", in: json) {
            return text
        }
        return nil
    }

    private nonisolated static func fastExtractJSONStringField(_ key: String, in json: String) -> String? {
        let quotedKey = "\"\(key)\""
        var searchStart = json.startIndex
        while let keyRange = json.range(of: quotedKey, range: searchStart..<json.endIndex) {
            var cursor = keyRange.upperBound
            skipJSONWhitespace(in: json, cursor: &cursor)
            guard cursor < json.endIndex, json[cursor] == ":" else {
                searchStart = keyRange.upperBound
                continue
            }
            cursor = json.index(after: cursor)
            skipJSONWhitespace(in: json, cursor: &cursor)
            guard cursor < json.endIndex, json[cursor] == "\"" else {
                searchStart = keyRange.upperBound
                continue
            }

            let valueQuoteStart = cursor
            let valueStart = json.index(after: cursor)
            cursor = valueStart
            var isEscaped = false
            var hasEscapes = false
            while cursor < json.endIndex {
                let char = json[cursor]
                if isEscaped {
                    isEscaped = false
                } else if char == "\\" {
                    isEscaped = true
                    hasEscapes = true
                } else if char == "\"" {
                    if !hasEscapes {
                        return String(json[valueStart..<cursor])
                    }
                    return decodeJSONStringLiteral(json[valueQuoteStart...cursor])
                }
                cursor = json.index(after: cursor)
            }
            return nil
        }
        return nil
    }

    private nonisolated static func skipJSONWhitespace(in json: String, cursor: inout String.Index) {
        while cursor < json.endIndex {
            switch json[cursor] {
            case " ", "\n", "\r", "\t":
                cursor = json.index(after: cursor)
            default:
                return
            }
        }
    }

    private nonisolated static func decodeJSONStringLiteral(_ literal: Substring) -> String? {
        let payload = #"{"value":"# + literal + "}"
        guard let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return nil
        }
        return obj["value"]
    }

    private nonisolated static func jsonStringLiteral(_ text: String) -> String {
        var output = "\""
        output.reserveCapacity(text.count + 2)
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x22:
                output += "\\\""
            case 0x5C:
                output += "\\\\"
            case 0x08:
                output += "\\b"
            case 0x0C:
                output += "\\f"
            case 0x0A:
                output += "\\n"
            case 0x0D:
                output += "\\r"
            case 0x09:
                output += "\\t"
            case 0x00...0x1F:
                output += String(format: "\\u%04X", scalar.value)
            default:
                output.unicodeScalars.append(scalar)
            }
        }
        output += "\""
        return output
    }
}

// MARK: - Stream Callback State

private final class StreamCallbackState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore

    init(continuation: AsyncThrowingStream<String, Error>.Continuation,
         doneSemaphore: DispatchSemaphore) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
    }
}

private final class ToolStreamCallbackState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<ConversationTurn, Error>.Continuation
    let doneSemaphore: DispatchSemaphore

    init(continuation: AsyncThrowingStream<ConversationTurn, Error>.Continuation,
         doneSemaphore: DispatchSemaphore) {
        self.continuation = continuation
        self.doneSemaphore = doneSemaphore
    }
}

// MARK: - Errors

public enum LiteRTLMError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case engineCreationFailed(String)
    case featureUnavailable(String)
    case inferenceFailure(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "LiteRT-LM model file not found"
        case .modelNotLoaded:
            "LiteRT-LM model is not loaded — call load() first"
        case .engineCreationFailed(let detail):
            "Failed to create LiteRT-LM engine: \(detail)"
        case .featureUnavailable(let detail):
            "LiteRT-LM feature unavailable: \(detail)"
        case .inferenceFailure(let detail):
            "LiteRT-LM inference failed: \(detail)"
        }
    }
}
