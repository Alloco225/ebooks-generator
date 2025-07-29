# API Setup Guide

## LLM Provider API Keys

This ebook generator supports multiple LLM providers. You need at least one API key to use the application.

### 1. OpenAI (GPT models + DALL-E image generation)
- **Models**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Features**: Text generation + Image generation
- **Get API Key**: https://platform.openai.com/api-keys
- **Environment Variable**: `OPENAI_API_KEY=sk-your-openai-api-key-here`

### 2. Anthropic (Claude models)
- **Models**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **Features**: Text generation only
- **Get API Key**: https://console.anthropic.com/
- **Environment Variable**: `ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here`

### 3. Google Gemini
- **Models**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini Pro
- **Features**: Text generation only
- **Get API Key**: https://aistudio.google.com/app/apikey
- **Environment Variable**: `GEMINI_API_KEY=your-gemini-api-key-here`

## Setup Instructions

1. Create a `.env` file in the project root
2. Add your API keys to the file:

```env
# Add only the providers you want to use
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

3. Install dependencies: `npm install`
4. Run the application: `npm run dev`

## Notes

- **Image Generation**: Currently only supported with OpenAI (DALL-E)
- **Minimum Requirement**: At least one API key must be set
- **Cost Considerations**: Different providers have different pricing models
- **Model Selection**: You can choose the specific model when running the application

## Provider Recommendations

- **OpenAI**: Best for full features (text + images), widely supported
- **Anthropic**: Excellent for text generation, good reasoning capabilities
- **Gemini**: Google's offering, competitive pricing and performance 