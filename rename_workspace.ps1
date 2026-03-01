# Run this script AFTER closing Cursor to complete the workspace restructure.
# 1. Removes the leftover LoRa_project_results folder (tools duplicate)
# 2. Renames Lora_results to LoRa_project_results

$root = "c:\Users\Ruben\Documents\VScode\Lora_results"
$parent = "c:\Users\Ruben\Documents\VScode"
$newName = "LoRa_project_results"

# Remove nested LoRa_project_results (duplicate tools - already at root)
$nested = Join-Path $root "LoRa_project_results"
if (Test-Path $nested) {
    Remove-Item $nested -Recurse -Force
    Write-Host "Removed nested LoRa_project_results folder"
}

# Rename workspace folder
$newPath = Join-Path $parent $newName
if ((Test-Path $root) -and -not (Test-Path $newPath)) {
    Rename-Item $root $newName
    Write-Host "Renamed Lora_results to LoRa_project_results"
    Write-Host "Reopen workspace from: $newPath"
} else {
    Write-Host "Rename skipped (target exists or source missing)"
}
