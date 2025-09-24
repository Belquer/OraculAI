# GitHub Secrets Configuration

This repository requires the following secrets to be configured in GitHub Settings > Secrets and variables > Actions:

## Required for Deployment
- `OPENAI_API_KEY` - OpenAI API key for embeddings and LLM
- `PINECONE_API_KEY` - Pinecone vector database API key  
- `PINECONE_ENVIRONMENT` - Pinecone environment (e.g., "us-east-1")

## Optional for Advanced Deployments
- `RENDER_API_KEY` - Render.com API key for automated deployments
- `RENDER_SERVICE_ID` - Render service ID for the OraculAI app

## Notes
- Workflows will run without these secrets but may skip certain features
- Docker builds will work but won't include pre-built vector indices
- Local development doesn't require these secrets if using DEV mode

## Setting up Secrets
1. Go to GitHub repository Settings
2. Navigate to "Secrets and variables" > "Actions"  
3. Click "New repository secret"
4. Add each required secret with its value