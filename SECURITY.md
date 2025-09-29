# Security Notice

## ⚠️ IMPORTANT SECURITY INFORMATION

This repository contains a template for analyzing ChatGPT conversation history. Before using this tool with your own data:

### 🔒 Security Checklist

- [ ] **API Key**: Replace `sk-proj-YOUR_ACTUAL_API_KEY_HERE` in `config/analysis_config.py` with your actual OpenAI API key
- [ ] **File Paths**: Update `CHAT_EXPORT_FILE` path to point to your actual chat export
- [ ] **Project Names**: Replace sample project names in `TARGET_PROJECTS` with your actual projects
- [ ] **Environment**: Set up a virtual environment and install dependencies
- [ ] **Permissions**: Ensure you have appropriate permissions to process your chat data

### 🚨 Never Commit Sensitive Data

- **API Keys**: Never commit real API keys to version control
- **Personal Data**: Never commit actual chat exports or analysis results
- **File Paths**: Use relative paths or environment variables for personal file locations

### 📁 Protected Files

The following files are protected by `.gitignore` and should never be committed:
- `data/raw/*.json` - Your actual chat exports
- `staged_results/` - Analysis results
- `knowledge_base/*.json` - Knowledge base data
- `batch_progress.json` - Processing progress
- `.env` - Environment variables

### 🔧 Setup Instructions

1. Copy `config/analysis_config.py` to `config/analysis_config_local.py`
2. Update the local config with your actual values
3. Use environment variables for sensitive data when possible
4. Test with sample data before using your actual chat history

### 📞 Support

If you encounter security issues or need help with setup, please:
1. Check the troubleshooting section in the README
2. Review the configuration validation
3. Test with sample data first

---

**Remember**: This tool processes your personal ChatGPT conversation data. Ensure you have appropriate permissions and consider privacy implications before sharing or storing analysis results.
