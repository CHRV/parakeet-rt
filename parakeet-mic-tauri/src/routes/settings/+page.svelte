<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";

    // AppConfig interface matching Rust backend
    interface AppConfig {
        encoder_model_path: string;
        decoder_model_path: string;
        vocab_path: string;
        ort_num_threads: number;
        left_context_size: number;
        chunk_size: number;
        right_context_size: number;
    }

    // State variables
    let config: AppConfig = $state({
        encoder_model_path: "",
        decoder_model_path: "",
        vocab_path: "",
        ort_num_threads: 4,
        left_context_size: 1.0,
        chunk_size: 0.25,
        right_context_size: 0.25,
    });

    let loading = $state(true);
    let saving = $state(false);
    let errorMessage: string | null = $state(null);
    let successMessage: string | null = $state(null);
    let validationErrors: Record<string, string> = $state({});

    // Validate settings before saving
    function validateSettings(): boolean {
        const errors: Record<string, string> = {};

        if (!config.encoder_model_path.trim()) {
            errors.encoder_model_path = "Encoder model path is required";
        }

        if (!config.decoder_model_path.trim()) {
            errors.decoder_model_path = "Decoder model path is required";
        }

        if (!config.vocab_path.trim()) {
            errors.vocab_path = "Vocabulary path is required";
        }

        if (config.ort_num_threads < 1 || config.ort_num_threads > 16) {
            errors.ort_num_threads = "Thread count must be between 1 and 16";
        }

        if (config.left_context_size < 0 || config.left_context_size > 10) {
            errors.left_context_size =
                "Left context size must be between 0 and 10 seconds";
        }

        if (config.chunk_size < 0.01 || config.chunk_size > 5) {
            errors.chunk_size = "Chunk size must be between 0.01 and 5 seconds";
        }

        if (config.right_context_size < 0 || config.right_context_size > 5) {
            errors.right_context_size =
                "Right context size must be between 0 and 5 seconds";
        }

        validationErrors = errors;
        return Object.keys(errors).length === 0;
    }

    // Load settings on mount
    onMount(async () => {
        try {
            const loadedConfig = await invoke<AppConfig>("get_settings");
            config = loadedConfig;
            loading = false;
        } catch (error) {
            errorMessage = `Failed to load settings: ${error}`;
            loading = false;
            console.error("Failed to load settings:", error);
        }
    });

    // Save settings handler
    async function handleSave() {
        // Clear previous messages
        errorMessage = null;
        successMessage = null;

        // Validate inputs
        if (!validateSettings()) {
            return;
        }

        saving = true;

        try {
            await invoke("save_settings", { config });
            successMessage = "Settings saved successfully!";

            // Clear success message after 3 seconds
            setTimeout(() => {
                successMessage = null;
            }, 3000);
        } catch (error) {
            errorMessage = `Failed to save settings: ${error}`;
            console.error("Failed to save settings:", error);
        } finally {
            saving = false;
        }
    }

    // Cancel handler - navigate back to main page
    function handleCancel() {
        window.location.href = "/";
    }
</script>

<div class="settings-container">
    <div class="settings-card">
        <div class="settings-header">
            <h1>⚙️ Settings</h1>
            <p class="subtitle">Configure transcription parameters</p>
        </div>

        {#if loading}
            <div class="loading-message">Loading settings...</div>
        {:else}
            <form
                class="settings-form"
                onsubmit={(e) => {
                    e.preventDefault();
                    handleSave();
                }}
            >
                <!-- Encoder Model Path -->
                <div class="form-group">
                    <label for="encoder_model_path">Encoder Model Path</label>
                    <input
                        type="text"
                        id="encoder_model_path"
                        bind:value={config.encoder_model_path}
                        class:error={validationErrors.encoder_model_path}
                        placeholder="models/encoder-model.int8.onnx"
                    />
                    {#if validationErrors.encoder_model_path}
                        <span class="error-message"
                            >{validationErrors.encoder_model_path}</span
                        >
                    {/if}
                </div>

                <!-- Decoder Model Path -->
                <div class="form-group">
                    <label for="decoder_model_path">Decoder Model Path</label>
                    <input
                        type="text"
                        id="decoder_model_path"
                        bind:value={config.decoder_model_path}
                        class:error={validationErrors.decoder_model_path}
                        placeholder="models/decoder_joint-model.int8.onnx"
                    />
                    {#if validationErrors.decoder_model_path}
                        <span class="error-message"
                            >{validationErrors.decoder_model_path}</span
                        >
                    {/if}
                </div>

                <!-- Vocabulary Path -->
                <div class="form-group">
                    <label for="vocab_path">Vocabulary Path</label>
                    <input
                        type="text"
                        id="vocab_path"
                        bind:value={config.vocab_path}
                        class:error={validationErrors.vocab_path}
                        placeholder="models/vocab.txt"
                    />
                    {#if validationErrors.vocab_path}
                        <span class="error-message"
                            >{validationErrors.vocab_path}</span
                        >
                    {/if}
                </div>

                <!-- ORT Thread Count -->
                <div class="form-group">
                    <label for="ort_num_threads"
                        >ORT Thread Count (1-16)
                        <span class="hint"
                            >Number of threads for model inference</span
                        >
                    </label>
                    <input
                        type="number"
                        id="ort_num_threads"
                        bind:value={config.ort_num_threads}
                        min="1"
                        max="16"
                        class:error={validationErrors.ort_num_threads}
                    />
                    {#if validationErrors.ort_num_threads}
                        <span class="error-message"
                            >{validationErrors.ort_num_threads}</span
                        >
                    {/if}
                </div>

                <!-- Left Context Size -->
                <div class="form-group">
                    <label for="left_context_size"
                        >Left Context Size (seconds)
                        <span class="hint"
                            >Amount of past audio context (0-10)</span
                        >
                    </label>
                    <input
                        type="number"
                        id="left_context_size"
                        bind:value={config.left_context_size}
                        min="0"
                        max="10"
                        step="0.1"
                        class:error={validationErrors.left_context_size}
                    />
                    {#if validationErrors.left_context_size}
                        <span class="error-message"
                            >{validationErrors.left_context_size}</span
                        >
                    {/if}
                </div>

                <!-- Chunk Size -->
                <div class="form-group">
                    <label for="chunk_size"
                        >Chunk Size (seconds)
                        <span class="hint"
                            >Audio processing chunk duration (0.01-5)</span
                        >
                    </label>
                    <input
                        type="number"
                        id="chunk_size"
                        bind:value={config.chunk_size}
                        min="0.01"
                        max="5"
                        step="0.01"
                        class:error={validationErrors.chunk_size}
                    />
                    {#if validationErrors.chunk_size}
                        <span class="error-message"
                            >{validationErrors.chunk_size}</span
                        >
                    {/if}
                </div>

                <!-- Right Context Size -->
                <div class="form-group">
                    <label for="right_context_size"
                        >Right Context Size (seconds)
                        <span class="hint"
                            >Amount of future audio context (0-5)</span
                        >
                    </label>
                    <input
                        type="number"
                        id="right_context_size"
                        bind:value={config.right_context_size}
                        min="0"
                        max="5"
                        step="0.1"
                        class:error={validationErrors.right_context_size}
                    />
                    {#if validationErrors.right_context_size}
                        <span class="error-message"
                            >{validationErrors.right_context_size}</span
                        >
                    {/if}
                </div>

                <!-- Success/Error Messages -->
                {#if successMessage}
                    <div class="message message-success">{successMessage}</div>
                {/if}

                {#if errorMessage}
                    <div class="message message-error">{errorMessage}</div>
                {/if}

                <!-- Action Buttons -->
                <div class="button-group">
                    <button
                        type="submit"
                        class="btn btn-save"
                        disabled={saving}
                    >
                        {saving ? "Saving..." : "Save Settings"}
                    </button>
                    <button
                        type="button"
                        class="btn btn-cancel"
                        onclick={handleCancel}
                        disabled={saving}
                    >
                        Cancel
                    </button>
                </div>
            </form>
        {/if}
    </div>
</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        background-color: #f5f5f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
            Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    .settings-container {
        width: 100vw;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        box-sizing: border-box;
    }

    .settings-card {
        width: 100%;
        max-width: 600px;
        background-color: #ffffff;
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .settings-header h1 {
        margin: 0 0 10px 0;
        font-size: 32px;
        color: #2c3e50;
    }

    .subtitle {
        margin: 0 0 30px 0;
        font-size: 16px;
        color: #95a5a6;
    }

    .loading-message {
        text-align: center;
        padding: 40px;
        color: #95a5a6;
        font-size: 18px;
    }

    .settings-form {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .form-group {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    label {
        font-size: 14px;
        font-weight: 500;
        color: #2c3e50;
    }

    .hint {
        font-size: 12px;
        font-weight: 400;
        color: #95a5a6;
        margin-left: 8px;
    }

    input[type="text"],
    input[type="number"] {
        padding: 12px 16px;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        font-size: 16px;
        font-family: inherit;
        transition: border-color 0.2s ease;
    }

    input[type="text"]:focus,
    input[type="number"]:focus {
        outline: none;
        border-color: #667eea;
    }

    input.error {
        border-color: #f56565;
    }

    .error-message {
        font-size: 13px;
        color: #f56565;
        margin-top: -4px;
    }

    .message {
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
    }

    .message-success {
        background-color: #eeffee;
        color: #33cc33;
    }

    .message-error {
        background-color: #ffeeee;
        color: #cc3333;
    }

    .button-group {
        display: flex;
        gap: 12px;
        margin-top: 10px;
    }

    .btn {
        flex: 1;
        padding: 14px 24px;
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

    .btn-save {
        background-color: #667eea;
    }

    .btn-save:hover:not(:disabled) {
        background-color: #5568d3;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transform: translateY(-1px);
    }

    .btn-cancel {
        background-color: #95a5a6;
    }

    .btn-cancel:hover:not(:disabled) {
        background-color: #7f8c8d;
        box-shadow: 0 4px 12px rgba(149, 165, 166, 0.4);
        transform: translateY(-1px);
    }
</style>
