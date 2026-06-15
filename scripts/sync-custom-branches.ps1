<#
.SYNOPSIS
Sync main from a remote, update custom feature branches, and optionally create
an integration branch that merges those custom branches.

.EXAMPLE
.\scripts\sync-custom-branches.ps1

.EXAMPLE
.\scripts\sync-custom-branches.ps1 -Remote upstream -UpstreamUrl https://github.com/dataelement/Clawith.git

.EXAMPLE
.\scripts\sync-custom-branches.ps1 -CustomBranches feature/external-http-channel-call,feature/custom-branding -CreateIntegrationBranch

.EXAMPLE
.\scripts\sync-custom-branches.ps1 -UpdateMode merge -CreateIntegrationBranch -IntegrationBranch release/custom-20260615 -Push
#>

[CmdletBinding()]
param(
    [string]$MainBranch = "main",

    [string]$Remote = "origin",

    [string]$UpstreamUrl = "",

    [string[]]$CustomBranches = @("feature/external-http-channel-call"),

    [ValidateSet("rebase", "merge")]
    [string]$UpdateMode = "rebase",

    [switch]$CreateIntegrationBranch,

    [string]$IntegrationBranch = ("release/custom-" + (Get-Date -Format "yyyyMMdd-HHmm")),

    [switch]$Push,

    [string]$PushRemote = "origin"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$GitArgs
    )

    Write-Host ("git " + ($GitArgs -join " ")) -ForegroundColor DarkGray
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($GitArgs -join ' ')"
    }
}

function Test-GitRef {
    param([string]$Ref)
    & git show-ref --verify --quiet $Ref
    return ($LASTEXITCODE -eq 0)
}

function Assert-CleanTrackedWorktree {
    $trackedStatus = & git status --porcelain=v1 -uno
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read git status."
    }

    if ($trackedStatus) {
        Write-Host "Tracked worktree changes detected:" -ForegroundColor Yellow
        $trackedStatus | ForEach-Object { Write-Host $_ }
        throw "Commit or stash tracked changes before syncing."
    }
}

function Ensure-Remote {
    param(
        [string]$Name,
        [string]$Url
    )

    $remotes = & git remote
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to list git remotes."
    }

    if ($remotes -contains $Name) {
        return
    }

    if ([string]::IsNullOrWhiteSpace($Url)) {
        throw "Remote '$Name' does not exist. Pass -UpstreamUrl to add it automatically."
    }

    Invoke-Git remote add $Name $Url
}

function Switch-ToBranch {
    param(
        [string]$Branch,
        [string]$TrackingRef = ""
    )

    if (Test-GitRef "refs/heads/$Branch") {
        Invoke-Git switch $Branch
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($TrackingRef)) {
        Invoke-Git switch -c $Branch --track $TrackingRef
        return
    }

    throw "Local branch '$Branch' does not exist."
}

Write-Host "== Clawith custom branch sync ==" -ForegroundColor Cyan
Invoke-Git rev-parse --show-toplevel | Out-Null

Assert-CleanTrackedWorktree
Ensure-Remote -Name $Remote -Url $UpstreamUrl

Write-Host "Fetching '$Remote'..." -ForegroundColor Cyan
Invoke-Git fetch --prune $Remote

$remoteMain = "$Remote/$MainBranch"
if (-not (Test-GitRef "refs/remotes/$remoteMain")) {
    throw "Remote branch '$remoteMain' was not found after fetch."
}

Write-Host "Updating '$MainBranch' from '$remoteMain' using fast-forward only..." -ForegroundColor Cyan
Switch-ToBranch -Branch $MainBranch -TrackingRef $remoteMain
Invoke-Git merge --ff-only $remoteMain

foreach ($branch in $CustomBranches) {
    if ([string]::IsNullOrWhiteSpace($branch)) {
        continue
    }

    Write-Host "Updating custom branch '$branch' from '$MainBranch' with $UpdateMode..." -ForegroundColor Cyan
    Switch-ToBranch -Branch $branch

    if ($UpdateMode -eq "rebase") {
        Invoke-Git rebase $MainBranch
    }
    else {
        Invoke-Git merge --no-ff $MainBranch
    }

    if ($Push) {
        if ($UpdateMode -eq "rebase") {
            Invoke-Git push --force-with-lease $PushRemote $branch
        }
        else {
            Invoke-Git push $PushRemote $branch
        }
    }
}

if ($CreateIntegrationBranch) {
    Write-Host "Creating integration branch '$IntegrationBranch' from '$MainBranch'..." -ForegroundColor Cyan
    Switch-ToBranch -Branch $MainBranch

    if (Test-GitRef "refs/heads/$IntegrationBranch") {
        throw "Integration branch '$IntegrationBranch' already exists."
    }

    Invoke-Git switch -c $IntegrationBranch

    foreach ($branch in $CustomBranches) {
        if ([string]::IsNullOrWhiteSpace($branch)) {
            continue
        }
        Invoke-Git merge --no-ff $branch
    }

    if ($Push) {
        Invoke-Git push -u $PushRemote $IntegrationBranch
    }
}

Write-Host "Done." -ForegroundColor Green
