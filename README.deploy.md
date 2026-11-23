# Vercel deployment

Deploy this `Text Summarizer` folder as the Vercel project root.

Set one of these environment variable configurations:

- `HF_INFERENCE_API_URL`: direct Hugging Face inference endpoint URL
- `HF_MODEL_ID`: Hugging Face model ID; defaults to `xenarx/text-summarizer-t5`
- `HF_API_TOKEN`: optional bearer token for private or rate-limited endpoints

For local use, set these values in `.env`; the app loads it automatically. Set either
`HF_MODEL_ID` or `HF_INFERENCE_API_URL`. `HF_API_TOKEN` is optional. Do not commit
`.env` or put secrets in Vercel source files; configure Vercel environment variables
in the Vercel project settings.

To use the local model instead, install `torch` and `transformers` and keep the
`Project/saved_summary_model` directory in place.
