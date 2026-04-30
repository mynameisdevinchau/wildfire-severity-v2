$files = Get-ChildItem data/interim/gridmet_remaining_single_rows/*.csv

foreach ($file in $files) {
    $id = ($file.BaseName -replace '^remaining_row_\d+_', '')
    $outFile = "data/raw/gridmet_output/$id.csv"

    if (Test-Path $outFile) {
        Write-Host "Skipping already downloaded $id"
        continue
    }

    Write-Host "Running $($file.Name)"

    pygridmet coords $file.FullName `
      -v pr `
      -v tmmn `
      -v tmmx `
      -v rmin `
      -v rmax `
      -v vs `
      -v srad `
      -v bi `
      -v erc `
      -v fm100 `
      -v fm1000 `
      -v vpd `
      -v pet `
      -s data/raw/gridmet_output

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED FULL VARIABLE SET: $($file.Name)"
        Write-Host "Trying again without pet..."

        pygridmet coords $file.FullName `
          -v pr `
          -v tmmn `
          -v tmmx `
          -v rmin `
          -v rmax `
          -v vs `
          -v srad `
          -v bi `
          -v erc `
          -v fm100 `
          -v fm1000 `
          -v vpd `
          -s data/raw/gridmet_output

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED WITHOUT PET: $($file.Name)"
            Add-Content -Path data/interim/gridmet_failed_single_files.txt -Value $file.FullName
        }
    }
}