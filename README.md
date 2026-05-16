# LiteRTLM-Swift

Swift wrapper for running LiteRT-LM models on iOS.

This fork is intentionally **Conversation API first**. MiloFlow uses it for stateful local chat, multimodal turns, tool calling, eager preface prewarm, and runtime-probed conversation cloning.

## Supported Path

Use `LiteRTLMEngine` with the persistent Conversation APIs:

```swift
let engine = LiteRTLMEngine(modelPath: modelURL)
try await engine.load()

try await engine.openConversation(
    systemPrompt: "You are Milo, an on-device assistant.",
    historyJSON: restoredHistoryJSON,
    tools: sharedTools
)

let reply = try await engine.conversationSendText("Help me plan today.")
```

This path handles prompt formatting internally and supports:

- text turns
- image turns
- audio turns
- mixed multimodal turns
- tool calls and tool results
- system prompt, tools, and restored-history prewarm
- stored conversation branches and clone activation

The older one-shot and Session APIs still compile for compatibility and fallback flows, but they are not the recommended MiloFlow integration path because they require manual prompt formatting and do not support the full conversation state model.

## Requirements

- iOS 17.0+
- Xcode 16+
- iPhone 13 Pro or later for Gemma 4 E2B class models
- `com.apple.developer.kernel.increased-memory-limit`

## Engine Setup

```swift
let engine = LiteRTLMEngine(
    modelPath: modelURL,
    backend: "cpu",
    enabledModalities: [.vision, .audio],
    maxNumImages: 4,
    enableBenchmarking: true
)

try await engine.load()
```

`backend: "gpu"` is available but should be treated as experimental on iOS. Text may use GPU when supported; vision/audio executor sections are still loaded conservatively.

## Conversation Turns

### Text

```swift
let text = try await engine.conversationSendText("Summarize this project.")
```

### Streaming Text

```swift
for try await chunk in engine.conversationSendTextStreaming("Write a short update.") {
    print(chunk, terminator: "")
}
```

### Images

```swift
let answer = try await engine.conversationSendImage(
    imageData: jpegData,
    prompt: "What is important in this image?",
    visualTokenBudget: 512
)
```

### Audio

```swift
let transcript = try await engine.conversationSendAudio(
    audioData: wavData,
    prompt: "Transcribe and summarize this audio.",
    format: .wav
)
```

### Mixed Multimodal

```swift
let response = try await engine.conversationSendMultimodal(
    audioData: [meetingAudio],
    imagesData: [whiteboardPhoto],
    prompt: "Connect the audio discussion with the whiteboard.",
    visualTokenBudget: 512
)
```

## Tool Calling

Open the conversation with tool declarations, then use the tool-aware turn APIs:

```swift
try await engine.openConversation(
    systemPrompt: systemPrompt,
    tools: [
        ToolDeclaration(
            name: "set_title",
            description: "Set the current workspace title.",
            parametersJSON: titleSchemaJSON
        )
    ]
)

let turn = try await engine.conversationSendTextWithTools("Name this workspace.")

switch turn {
case .text(let text):
    print(text)
case .toolCalls(let calls):
    let results = calls.map { call in
        ToolResult(toolName: call.name, contentJSON: #"{"ok":true}"#)
    }
    let followUp = try await engine.conversationSendToolResults(results)
    print(followUp)
}
```

Tool-enabled conversations set `filterChannelContentFromKVCache` so tool channel content does not pollute the reusable KV cache.

## Prewarm

`openConversation` and `openConversationBranch` opt into eager preface prefill:

```swift
litert_lm_conversation_config_set_prefill_preface_on_init(config, true)
```

That means system prompt, tools, and restored history are prefetched into the KV cache at conversation creation time. First user turn latency then pays only for the new user turn plus decode.

Official upstream LiteRT-LM already has the native C++ `ConversationConfig::Builder().SetPrefillPrefaceOnInit(...)` path. The C setter is added only while packaging the xcframework, so the upstream source checkout remains clean.

## Conversation Branching

Branching is automatic when supported by the loaded backend:

```swift
if engine.supportsConversationClone {
    try await engine.openConversationBranch(
        "base",
        systemPrompt: systemPrompt,
        historyJSON: restoredHistoryJSON,
        tools: sharedTools
    )

    try await engine.cloneConversationBranch("base", as: "project")
    try await engine.activateConversationBranch("project")
}
```

`supportsConversationClone` is probed once during `load()`. Current upstream exposes `litert_lm_conversation_clone`; supported `SessionAdvanced` loads can clone, while `SessionBasic` still returns unsupported. Callers must fall back to `openConversation` when the probe is false.

MiloFlow enables the branch path in app code and automatically disables it if the runtime probe fails.

## Public Surface

Primary API:

| API | Purpose |
| --- | --- |
| `load()` / `unload()` | Engine lifecycle |
| `openConversation(...)` | Open a stateful text/multimodal/tool conversation |
| `closeConversation()` / `closeConversationAndWait()` | Release persistent conversation state |
| `conversationSendText(...)` | Text turn |
| `conversationSendTextStreaming(...)` | Streaming text turn |
| `conversationSendTextWithTools(...)` | Tool-aware text turn |
| `conversationSendToolResults(...)` | Return tool results |
| `conversationSendImage(...)` / `conversationSendImages(...)` | Image turns |
| `conversationSendAudio(...)` | Audio turn |
| `conversationSendMultimodal(...)` | Mixed audio/image turn |
| `cancelActiveConversationProcess()` | Cancel active conversation work |
| `openConversationBranch(...)` | Create a prewarmed stored branch |
| `saveConversationBranch(...)` | Clone current conversation into a stored branch |
| `cloneConversationBranch(_:as:)` | Clone stored branch to another stored branch |
| `activateConversationBranch(...)` | Clone stored branch into the active conversation slot |
| `supportsConversationClone` | Runtime clone capability probe |
| `benchmarkSnapshot()` | Latest native LiteRT benchmark counters for diagnostics |

Compatibility API:

| API | Replacement |
| --- | --- |
| `generate(...)`, `generateStreaming(...)` | `openConversation` + `conversationSendText...` |
| `vision(...)`, `visionMultiImage(...)` | `conversationSendImage...` |
| `audio(...)` | `conversationSendAudio...` |
| `multimodal(...)` | `conversationSendMultimodal...` |
| `openSession(...)`, `sessionGenerateStreaming(...)`, `sessionPrefill(...)` | Conversation APIs |
| `tokenize(...)`, `detokenize(...)` | Diagnostics only |
| `ModelDownloader` | Demo/helper only; MiloFlow owns its model asset flow |

## Building The XCFramework

Use the script:

```bash
./scripts/build-xcframework.sh /path/to/LiteRT-LM
```

When the input checkout is official upstream, the script temporarily injects:

- `//c:libLiteRTLMEngine.dylib`
- `litert_lm_conversation_config_set_prefill_preface_on_init`

The script restores `c/BUILD`, `c/engine.cc`, and `c/engine.h` before exit. This keeps `ThirdParty/LiteRT-LM` merge-friendly while producing the iOS framework MiloFlow needs.

Verify exported symbols:

```bash
nm -gU Frameworks/LiteRTLM.xcframework/ios-arm64/CLiteRTLM.framework/CLiteRTLM \
  | rg "litert_lm_conversation_clone|litert_lm_conversation_config_set_prefill_preface_on_init"
```

Expected:

```text
_litert_lm_conversation_clone
_litert_lm_conversation_config_set_prefill_preface_on_init
```

## Notes

- All native C API calls are serialized on an internal dispatch queue.
- The wrapper is `@unchecked Sendable` because native handles are confined to that queue.
- The included xcframework contains Google LiteRT-LM code under Apache 2.0.
- This Swift wrapper is a community integration, not an official Google product.

## License

MIT License. See `LICENSE`.
