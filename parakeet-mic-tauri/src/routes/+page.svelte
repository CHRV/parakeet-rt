<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { onMount } from "svelte";
  import { writable, derived } from "svelte/store";

  // State variables using stores
  type RecordingState = "idle" | "recording" | "error";

  const state = writable<RecordingState>("idle");
  const transcript = writable<string>("");
  const errorMessage = writable<string | null>(null);
  const audioLevel = writable<number>(0);

  // Canvas reference for visualizer
  let canvasElement: HTMLCanvasElement;

  // Transcript area reference for auto-scroll
  let transcriptAreaElement: HTMLDivElement;

  // Event listener cleanup functions
  let unlistenTranscription: UnlistenFn | null = null;
  let unlistenAudioLevel: UnlistenFn | null = null;
  let unlistenError: UnlistenFn | null = null;

  // Format transcript with line breaks after sentence-ending punctuation
  function formatTranscript(text: string): string {
    return text;
    //return text.replace(/([.!?])\s+/g, "$1\n").trim();
  }

  // Derived formatted transcript
  const formattedTranscript = derived(transcript, ($transcript) =>
    formatTranscript($transcript)
  );

  // Auto-scroll when transcript changes
  $: if ($transcript && transcriptAreaElement) {
    transcriptAreaElement.scrollTop = transcriptAreaElement.scrollHeight;
  }

  // Audio visualizer reactive statement
  $: if (canvasElement) {
    const ctx = canvasElement.getContext("2d");
    if (ctx) {
      // Clear canvas
      ctx.fillStyle = "#f7fafc";
      ctx.fillRect(0, 0, 280, 50);

      if ($state === "recording" && $audioLevel > 0) {
        // Draw waveform based on audio level
        const centerY = 25;
        const barWidth = 4;
        const barSpacing = 2;
        const numBars = Math.floor(280 / (barWidth + barSpacing));

        ctx.fillStyle = "#667eea";

        for (let i = 0; i < numBars; i++) {
          // Create a wave pattern with some randomness based on audio level
          const variation = Math.sin(i * 0.5 + Date.now() * 0.005) * 0.3 + 0.7;
          const barHeight = $audioLevel * 40 * variation;
          const x = i * (barWidth + barSpacing);
          const y = centerY - barHeight / 2;

          ctx.fillRect(x, y, barWidth, barHeight);
        }
      } else {
        // Show empty state - flat line
        ctx.strokeStyle = "#cbd5e0";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 25);
        ctx.lineTo(280, 25);
        ctx.stroke();
      }
    }
  }

  // Control button handlers
  async function startRecording() {
    try {
      state.set("recording");
      errorMessage.set(null);
      await invoke("start_recording");
    } catch (error) {
      state.set("error");
      errorMessage.set(String(error));
      console.error("Failed to start recording:", error);
    }
  }

  async function stopRecording() {
    try {
      await invoke("stop_recording");
      state.set("idle");
    } catch (error) {
      state.set("error");
      errorMessage.set(String(error));
      console.error("Failed to stop recording:", error);
    }
  }

  function navigateToSettings() {
    window.location.href = "/settings";
  }

  onMount(() => {
    // Set up event listeners
    (async () => {
      try {
        // Listen for transcription events
        unlistenTranscription = await listen<string>(
          "transcription",
          (event) => {
            transcript.update((t) => t + event.payload);
          }
        );

        // Listen for audio level events
        unlistenAudioLevel = await listen<number>("audio-level", (event) => {
          audioLevel.set(event.payload);
        });

        // Listen for error events
        unlistenError = await listen<string>("error", (event) => {
          state.set("error");
          errorMessage.set(event.payload);
          console.error("Backend error:", event.payload);
        });
      } catch (error) {
        console.error("Failed to set up event listeners:", error);
      }
    })();

    return () => {
      // Cleanup event listeners
      if (unlistenTranscription) unlistenTranscription();
      if (unlistenAudioLevel) unlistenAudioLevel();
      if (unlistenError) unlistenError();
    };
  });
</script>

<div class="app-container">
  <!-- Full-page transcript display area -->
  <div class="transcript-area" bind:this={transcriptAreaElement}>
    <div class="transcript-content">
      {#if $transcript}
        {$transcript}
      {:else}
        <p class="placeholder">Waiting for transcription...</p>
      {/if}
    </div>
  </div>

  <!-- Floating control panel -->
  <div class="control-panel">
    <div class="panel-header">
      <h2>🎤 Parakeet</h2>
      <p class="subtitle">Real-time transcription</p>
    </div>

    <!-- Status indicator -->
    <div class="status-indicator status-{$state}">
      {#if $state === "idle"}
        Ready
      {:else if $state === "recording"}
        <span class="pulse-dot"></span> Recording
      {:else if $state === "error"}
        Error: {$errorMessage || "Unknown error"}
      {/if}
    </div>

    <!-- Audio visualizer -->
    <div class="visualizer-container">
      <canvas bind:this={canvasElement} width="280" height="50"></canvas>
    </div>

    <!-- Control buttons -->
    <div class="button-group">
      <button
        class="btn btn-start"
        disabled={$state === "recording"}
        onclick={startRecording}
      >
        Start
      </button>
      <button
        class="btn btn-stop"
        disabled={$state !== "recording"}
        onclick={stopRecording}
      >
        Stop
      </button>
    </div>

    <button class="btn btn-settings" onclick={navigateToSettings}>
      ⚙️ Settings
    </button>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background-color: #f5f5f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
      Ubuntu, Cantarell, sans-serif;
  }

  .app-container {
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    position: relative;
  }

  /* Transcript display area */
  .transcript-area {
    width: 100%;
    height: 100%;
    overflow-y: auto;
    padding: 80px 120px;
    box-sizing: border-box;
  }

  .transcript-content {
    max-width: 800px;
    margin: 0 auto;
    background-color: #ffffff;
    padding: 40px;
    min-height: 200px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .transcript-content {
    font-family: Georgia, serif;
    font-size: 20px;
    line-height: 1.8;
    color: #2c3e50;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .placeholder {
    color: #95a5a6;
    font-style: italic;
    text-align: center;
    margin: 0;
  }

  /* Floating control panel */
  .control-panel {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 320px;
    background-color: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    z-index: 1000;
  }

  .panel-header h2 {
    margin: 0 0 5px 0;
    font-size: 24px;
    color: #2c3e50;
  }

  .subtitle {
    margin: 0 0 15px 0;
    font-size: 14px;
    color: #95a5a6;
  }

  /* Status indicator */
  .status-indicator {
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 15px;
    font-weight: 500;
    text-align: center;
  }

  .status-idle {
    background-color: #eeffee;
    color: #33cc33;
  }

  .status-recording {
    background-color: #fef3cd;
    color: #856404;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .status-error {
    background-color: #ffeeee;
    color: #cc3333;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #f56565;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  /* Visualizer container */
  .visualizer-container {
    margin-bottom: 15px;
    background-color: #f7fafc;
    border-radius: 8px;
    overflow: hidden;
  }

  canvas {
    display: block;
    width: 100%;
    height: 50px;
  }

  /* Buttons */
  .button-group {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
  }

  .btn {
    flex: 1;
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    color: white;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-start {
    background-color: #667eea;
  }

  .btn-start:hover:not(:disabled) {
    background-color: #5568d3;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    transform: translateY(-1px);
  }

  .btn-stop {
    background-color: #f56565;
  }

  .btn-stop:hover:not(:disabled) {
    background-color: #e53e3e;
    box-shadow: 0 4px 12px rgba(245, 101, 101, 0.4);
    transform: translateY(-1px);
  }

  .btn-settings {
    width: 100%;
    background-color: #95a5a6;
  }

  .btn-settings:hover {
    background-color: #7f8c8d;
    box-shadow: 0 4px 12px rgba(149, 165, 166, 0.4);
    transform: translateY(-1px);
  }
</style>
