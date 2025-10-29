# PowerShell script to download Llama models from Meta
# Make sure you have your custom download URL ready from Meta

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Llama Model Download Helper" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if llama-stack is installed
Write-Host "Step 1: Installing llama-stack CLI..." -ForegroundColor Yellow
pip install llama-stack -U

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Step 2: Listing available models..." -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
& "$env:APPDATA\Python\Python312\Scripts\llama.exe" model list

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Recommended models for fine-tuning:" -ForegroundColor Green
Write-Host "  - Llama-3.1-8B-Instruct      (best for 24GB GPU)" -ForegroundColor White
Write-Host "  - Llama-3.2-3B-Instruct      (works on 16GB GPU)" -ForegroundColor White
Write-Host "  - Llama-3.2-1B-Instruct      (works on 8GB GPU)" -ForegroundColor White
Write-Host "  - Llama-2-7b-chat-hf         (Llama 2 for 24GB GPU)" -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Prompt user for model ID
$modelId = Read-Host "Enter the model ID you want to download (e.g., Llama-3.1-8B-Instruct)"

if ([string]::IsNullOrWhiteSpace($modelId)) {
    Write-Host "No model ID provided. Exiting." -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Step 3: Downloading $modelId..." -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "When prompted, paste your custom URL from Meta's download page" -ForegroundColor Yellow
Write-Host "Your URL should start with: https://download.llamameta.net/..." -ForegroundColor Yellow
Write-Host ""

# Run the download command
C:/Python312/python.exe -m llama_stack.cli.llama model download --source meta --model-id $modelId

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Download complete!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Default model location: $env:USERPROFILE\.llama\checkpoints\$modelId" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update the notebook cell with your model path" -ForegroundColor White
Write-Host "2. Run the fine-tuning notebook cells in order" -ForegroundColor White
Write-Host ""
