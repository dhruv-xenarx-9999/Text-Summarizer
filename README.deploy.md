# Vercel deployment

Deploy this `Text Summarizer` folder as the Vercel project root.

Set one of these environment variable configurations:

- `HF_INFERENCE_API_URL`: direct Hugging Face inference endpoint URL
- `HF_MODEL_ID`: Hugging Face model ID; the app constructs the endpoint URL
- `HF_API_TOKEN`: optional bearer token for private or rate-limited endpoints

For local use, install `torch` and `transformers` and keep the `Project/saved_summary_model` directory in place.
