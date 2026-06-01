#!/usr/bin/env node
import { execFile } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
const execFileAsync = promisify(execFile);
export const DEFAULT_TIMEOUT_SECONDS = 20 * 60;
export const DEFAULT_EYES_TIMEOUT_SECONDS = 60;
export const DEFAULT_POLL_SECONDS = 30;
export const PREVIEW_MAX_LENGTH = 220;
export const REVIEW_TRIGGER_BOT = "chatgpt-codex-connector[bot]";
export function normalizeBody(body = "") {
    return body.replace(/\s+/g, " ").trim();
}
function sanitizePreviewParagraph(paragraph = "") {
    return normalizeBody(paragraph
        .replace(/!\[[^\]]*]\([^)]*\)/g, "")
        .replace(/<\/?[^>]+>/g, " ")
        .replace(/^#+\s*/gm, "")
        .replace(/\*\*/g, ""));
}
export function extractTitle(body = "") {
    const withoutDetails = body.replace(/<details[\s\S]*$/i, "");
    const title = withoutDetails
        .split(/\n\s*\n/)
        .map((paragraph) => sanitizePreviewParagraph(paragraph))
        .find((paragraph) => paragraph.length > 0);
    return title && title.length > 0 ? title : null;
}
export function extractSeverity(body = "") {
    const match = body.match(/\b(P[0-3])\s+Badge\b/i) ?? body.match(/\b(P[0-3])\b/);
    return match ? match[1].toUpperCase() : null;
}
export function extractPreview(body = "", maxLength = PREVIEW_MAX_LENGTH) {
    const withoutDetails = body.replace(/<details[\s\S]*$/i, "");
    const paragraphs = withoutDetails
        .split(/\n\s*\n/)
        .map((paragraph) => sanitizePreviewParagraph(paragraph))
        .filter((paragraph) => paragraph.length > 0);
    let normalized = paragraphs[0] ?? sanitizePreviewParagraph(withoutDetails);
    let paragraphIndex = 1;
    while (normalized.length > 0 &&
        normalized.length < Math.min(80, maxLength) &&
        paragraphIndex < paragraphs.length) {
        normalized = `${normalized} ${paragraphs[paragraphIndex]}`;
        paragraphIndex += 1;
    }
    if (normalized.length === 0) {
        return "(empty)";
    }
    if (normalized.length <= maxLength) {
        return normalized;
    }
    return `${normalized.slice(0, maxLength - 1)}…`;
}
export function isReviewTriggerBody(body = "") {
    return /^@codex review\b/i.test(normalizeBody(body));
}
export function hasEyesReaction(comment) {
    return Number(comment?.reactions?.eyes ?? 0) > 0;
}
export function classifyBotComment(body = "") {
    const lowered = normalizeBody(body).toLowerCase();
    if (lowered.includes("usage limit") ||
        lowered.includes("usage limits") ||
        lowered.includes("usage dashboard")) {
        return "bot_quota";
    }
    if (lowered.includes("failed") ||
        lowered.includes("error") ||
        lowered.includes("unable") ||
        lowered.includes("could not") ||
        lowered.includes("couldn't")) {
        return "bot_error";
    }
    return "bot_message";
}
function toTimestamp(value) {
    const timestamp = Date.parse(value ?? "");
    return Number.isNaN(timestamp) ? 0 : timestamp;
}
function latestBy(items, getTimestamp) {
    if (items.length === 0) {
        return null;
    }
    return items.reduce((latest, current) => getTimestamp(current) >= getTimestamp(latest) ? current : latest);
}
function dedupeById(items) {
    const seen = new Set();
    return items.filter((item) => {
        if (seen.has(item.ref)) {
            return false;
        }
        seen.add(item.ref);
        return true;
    });
}
export function buildIssueCommentRef(owner, repo, commentId) {
    return `gh api repos/${owner}/${repo}/issues/comments/${commentId}`;
}
export function buildReviewRef(owner, repo, prNumber, reviewId) {
    return `gh api repos/${owner}/${repo}/pulls/${prNumber}/reviews/${reviewId}`;
}
export function buildPullCommentRef(owner, repo, commentId) {
    return `gh api repos/${owner}/${repo}/pulls/comments/${commentId}`;
}
export function findLatestTriggerComment(issueComments) {
    return latestBy(issueComments.filter((comment) => isReviewTriggerBody(comment.body)), (comment) => toTimestamp(comment.created_at));
}
export function findLatestIssueComment(issueComments) {
    return latestBy(issueComments, (comment) => toTimestamp(comment.created_at));
}
export function findLatestBotCommentAfterTrigger(issueComments, triggerComment) {
    const triggerTime = toTimestamp(triggerComment.created_at);
    return latestBy(issueComments.filter((comment) => comment.id !== triggerComment.id &&
        comment.user?.login === REVIEW_TRIGGER_BOT &&
        toTimestamp(comment.created_at) >= triggerTime), (comment) => toTimestamp(comment.created_at));
}
export function findLatestBotReviewAfterTrigger(reviews, triggerComment) {
    const triggerTime = toTimestamp(triggerComment.created_at);
    return latestBy(reviews.filter((review) => review.user?.login === REVIEW_TRIGGER_BOT &&
        toTimestamp(review.submitted_at) >= triggerTime), (review) => toTimestamp(review.submitted_at));
}
export function summarizeReviews(owner, repo, prNumber, reviews, triggerComment, currentResultReviewId = null) {
    const triggerTime = toTimestamp(triggerComment.created_at);
    return dedupeById(reviews
        .filter((review) => toTimestamp(review.submitted_at) >= triggerTime)
        .filter((review) => review.user?.login === REVIEW_TRIGGER_BOT)
        .filter((review) => review.id !== currentResultReviewId)
        .map((review) => ({
        ref: buildReviewRef(owner, repo, prNumber, review.id),
        link: review.html_url ?? null,
        state: review.state ?? "UNKNOWN",
        preview: normalizeBody(review.body ?? "").length > 0
            ? extractPreview(review.body)
            : null,
    })));
}
export function summarizeInlineComments(owner, repo, pullComments, triggerComment, currentResultReviewId = null) {
    const triggerTime = toTimestamp(triggerComment.created_at);
    return dedupeById(pullComments
        .filter((comment) => toTimestamp(comment.created_at) >= triggerTime)
        .filter((comment) => comment.user?.login === REVIEW_TRIGGER_BOT)
        .filter((comment) => comment.in_reply_to_id == null)
        .filter((comment) => currentResultReviewId == null
        ? true
        : comment.pull_request_review_id === currentResultReviewId)
        .map((comment) => ({
        ref: buildPullCommentRef(owner, repo, comment.id),
        link: comment.html_url ?? null,
        location: comment.path && comment.line != null
            ? `${comment.path}:${comment.line}`
            : (comment.path ?? null),
        severity: extractSeverity(comment.body ?? ""),
        title: extractTitle(comment.body ?? ""),
        preview: extractPreview(comment.body ?? ""),
    })));
}
export function deriveWaitState(input) {
    const latestTrigger = findLatestTriggerComment(input.issueComments);
    if (!latestTrigger) {
        throw new Error("Could not find an active `@codex review` trigger comment.");
    }
    const previousActive = input.activeTriggerId
        ? input.issueComments.find((comment) => String(comment.id) === String(input.activeTriggerId)) ?? latestTrigger
        : latestTrigger;
    const activeChanged = String(previousActive.id) !== String(latestTrigger.id);
    const activeTrigger = activeChanged ? latestTrigger : previousActive;
    const latestIssueComment = findLatestIssueComment(input.issueComments);
    const latestBotComment = findLatestBotCommentAfterTrigger(input.issueComments, activeTrigger);
    const latestBotReview = findLatestBotReviewAfterTrigger(input.reviews, activeTrigger);
    const latestBotCommentTime = toTimestamp(latestBotComment?.created_at);
    const latestBotReviewTime = toTimestamp(latestBotReview?.submitted_at);
    const resultComment = latestBotComment && latestBotCommentTime >= latestBotReviewTime
        ? latestBotComment
        : null;
    const resultReview = latestBotReview && latestBotReviewTime > latestBotCommentTime
        ? latestBotReview
        : null;
    const resultKind = resultComment ? classifyBotComment(resultComment.body) : null;
    const elapsedMs = Math.max(0, input.nowMs - input.startedAtMs);
    const triggerAgeMs = Math.max(0, input.nowMs - toTimestamp(activeTrigger.created_at));
    const liveEyesSeen = hasEyesReaction(activeTrigger);
    const eyesSeen = activeChanged
        ? liveEyesSeen
        : input.eyesSeenEverForActiveTrigger || liveEyesSeen;
    const eyesLate = !eyesSeen && triggerAgeMs >= input.eyesTimeoutMs;
    return {
        status: resultComment || resultReview
            ? "done"
            : elapsedMs >= input.timeoutMs
                ? "timeout"
                : "waiting",
        elapsedMs,
        commentCount: input.issueComments.length,
        activeTrigger,
        activeChanged,
        eyesSeen,
        eyesLate,
        nextEyesSeenEver: eyesSeen,
        latestIssueComment,
        latestBotComment,
        latestBotReview,
        resultComment,
        resultReview,
        resultKind,
    };
}
export function formatElapsed(elapsedMs) {
    const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
export function formatProgressLine(owner, repo, prNumber, state) {
    const parts = [
        "waiting",
        `elapsed=${formatElapsed(state.elapsedMs)}`,
        `repo=${owner}/${repo}`,
        `pr=${prNumber}`,
        `comments=${state.commentCount}`,
        `eyes=${state.eyesSeen ? "1" : "0"}`,
        `active=${buildIssueCommentRef(owner, repo, state.activeTrigger.id)}`,
    ];
    if (state.latestIssueComment &&
        String(state.latestIssueComment.id) !== String(state.activeTrigger.id)) {
        parts.push(`last=${buildIssueCommentRef(owner, repo, state.latestIssueComment.id)}`);
    }
    return parts.join(" ");
}
function parseRepoAndPrFromArgument(value) {
    const hashIndex = value.lastIndexOf("#");
    if (hashIndex <= 0 || hashIndex === value.length - 1) {
        return { repo: null, pr: null };
    }
    return {
        repo: value.slice(0, hashIndex),
        pr: value.slice(hashIndex + 1),
    };
}
export function parseArgs(argv) {
    const options = {
        repo: null,
        pr: null,
        timeoutSeconds: DEFAULT_TIMEOUT_SECONDS,
        eyesTimeoutSeconds: DEFAULT_EYES_TIMEOUT_SECONDS,
        pollSeconds: DEFAULT_POLL_SECONDS,
    };
    for (let index = 0; index < argv.length; index += 1) {
        const arg = argv[index];
        if (arg === "-R" || arg === "--repo") {
            options.repo = argv[index + 1] ?? null;
            index += 1;
            continue;
        }
        if (arg === "--timeout") {
            options.timeoutSeconds = Number(argv[index + 1] ?? DEFAULT_TIMEOUT_SECONDS);
            index += 1;
            continue;
        }
        if (arg === "--eyes-timeout") {
            options.eyesTimeoutSeconds = Number(argv[index + 1] ?? DEFAULT_EYES_TIMEOUT_SECONDS);
            index += 1;
            continue;
        }
        if (arg === "--interval") {
            options.pollSeconds = Number(argv[index + 1] ?? DEFAULT_POLL_SECONDS);
            index += 1;
            continue;
        }
        if (arg === "--help" || arg === "-h") {
            printUsage();
            process.exit(0);
        }
        if (arg.startsWith("-")) {
            throw new Error(`Unknown argument: ${arg}`);
        }
        if (options.pr !== null) {
            throw new Error(`Unexpected extra argument: ${arg}`);
        }
        const parsed = parseRepoAndPrFromArgument(arg);
        if (parsed.repo && parsed.pr) {
            options.repo ??= parsed.repo;
            options.pr = parsed.pr;
            continue;
        }
        options.pr = arg;
    }
    if (!options.pr) {
        throw new Error("Missing required pull request number.");
    }
    if (!Number.isFinite(options.timeoutSeconds) ||
        !Number.isFinite(options.eyesTimeoutSeconds) ||
        !Number.isFinite(options.pollSeconds)) {
        throw new Error("Timeout and interval arguments must be numbers.");
    }
    const pr = options.pr;
    return {
        repo: options.repo,
        pr,
        timeoutSeconds: options.timeoutSeconds,
        eyesTimeoutSeconds: options.eyesTimeoutSeconds,
        pollSeconds: options.pollSeconds,
    };
}
function printUsage() {
    console.log("Usage:");
    console.log("  node scripts/review-wait.mjs <owner/repo#pr>");
}
async function ghJson(args) {
    try {
        const { stdout } = await execFileAsync("gh", args, {
            maxBuffer: 10 * 1024 * 1024,
        });
        return JSON.parse(stdout);
    }
    catch (error) {
        const stderr = error && typeof error === "object" && "stderr" in error
            ? String(error.stderr ?? "").trim()
            : "";
        const message = stderr.length > 0
            ? stderr
            : error instanceof Error
                ? error.message
                : String(error);
        throw new Error(`gh command failed: ${message}`);
    }
}
async function ghText(args) {
    try {
        const { stdout } = await execFileAsync("gh", args, {
            maxBuffer: 1024 * 1024,
        });
        return String(stdout).trim();
    }
    catch (error) {
        const stderr = error && typeof error === "object" && "stderr" in error
            ? String(error.stderr ?? "").trim()
            : "";
        const message = stderr.length > 0
            ? stderr
            : error instanceof Error
                ? error.message
                : String(error);
        throw new Error(`gh command failed: ${message}`);
    }
}
async function resolveRepo(explicitRepo) {
    if (explicitRepo) {
        return explicitRepo;
    }
    const detectedRepo = await ghText([
        "repo",
        "view",
        "--json",
        "nameWithOwner",
        "--jq",
        ".nameWithOwner",
    ]);
    if (detectedRepo.length === 0) {
        throw new Error("Could not detect the current GitHub repository.");
    }
    return detectedRepo;
}
async function fetchIssueComments(owner, repo, prNumber) {
    return ghJson([
        "api",
        `repos/${owner}/${repo}/issues/${prNumber}/comments`,
        "--paginate",
    ]);
}
async function fetchReviews(owner, repo, prNumber) {
    return ghJson([
        "api",
        `repos/${owner}/${repo}/pulls/${prNumber}/reviews`,
        "--paginate",
    ]);
}
async function fetchPullComments(owner, repo, prNumber) {
    return ghJson([
        "api",
        `repos/${owner}/${repo}/pulls/${prNumber}/comments`,
        "--paginate",
    ]);
}
export async function waitForReviewActivity(parsedArgs) {
    const repoNameWithOwner = await resolveRepo(parsedArgs.repo);
    const [owner, repo] = repoNameWithOwner.split("/", 2);
    if (!owner || !repo) {
        throw new Error(`Invalid repo format: ${repoNameWithOwner}`);
    }
    const startedAtMs = Date.now();
    let activeTriggerId = null;
    let eyesSeenEverForActiveTrigger = false;
    while (true) {
        const [issueComments, reviews] = await Promise.all([
            fetchIssueComments(owner, repo, parsedArgs.pr),
            fetchReviews(owner, repo, parsedArgs.pr),
        ]);
        const state = deriveWaitState({
            issueComments,
            reviews,
            activeTriggerId,
            eyesSeenEverForActiveTrigger,
            startedAtMs,
            nowMs: Date.now(),
            timeoutMs: parsedArgs.timeoutSeconds * 1000,
            eyesTimeoutMs: parsedArgs.eyesTimeoutSeconds * 1000,
        });
        activeTriggerId = String(state.activeTrigger.id);
        eyesSeenEverForActiveTrigger = state.nextEyesSeenEver;
        if (state.status === "done") {
            const [finalReviews, pullComments] = await Promise.all([
                fetchReviews(owner, repo, parsedArgs.pr),
                fetchPullComments(owner, repo, parsedArgs.pr),
            ]);
            return {
                repo: `${owner}/${repo}`,
                pr: parsedArgs.pr,
                elapsedMs: state.elapsedMs,
                activeTrigger: state.activeTrigger,
                latestIssueComment: state.latestIssueComment,
                resultComment: state.resultComment,
                resultReview: state.resultReview,
                resultKind: state.resultKind,
                eyesSeen: state.eyesSeen,
                timedOut: false,
                reviewSummaries: summarizeReviews(owner, repo, parsedArgs.pr, finalReviews, state.activeTrigger, state.resultReview?.id ?? null),
                inlineSummaries: summarizeInlineComments(owner, repo, pullComments, state.activeTrigger, state.resultReview?.id ?? null),
            };
        }
        if (state.status === "timeout") {
            return {
                repo: `${owner}/${repo}`,
                pr: parsedArgs.pr,
                elapsedMs: state.elapsedMs,
                activeTrigger: state.activeTrigger,
                latestIssueComment: state.latestIssueComment,
                resultComment: state.resultComment,
                resultReview: state.resultReview,
                resultKind: state.resultKind,
                eyesSeen: state.eyesSeen,
                timedOut: true,
                reviewSummaries: [],
                inlineSummaries: [],
            };
        }
        console.log(formatProgressLine(owner, repo, parsedArgs.pr, state));
        await delay(parsedArgs.pollSeconds * 1000);
    }
}
function printResult(result) {
    const [owner, repo] = result.repo.split("/", 2);
    if (result.timedOut || (!result.resultComment && !result.resultReview)) {
        const parts = [
            "timeout",
            `elapsed=${formatElapsed(result.elapsedMs)}`,
            `repo=${result.repo}`,
            `pr=${result.pr}`,
            `eyes=${result.eyesSeen ? "1" : "0"}`,
            `active=${buildIssueCommentRef(owner, repo, result.activeTrigger.id)}`,
        ];
        if (result.latestIssueComment) {
            parts.push(`last=${buildIssueCommentRef(owner, repo, result.latestIssueComment.id)}`);
        }
        console.log(parts.join(" "));
        return;
    }
    const resultRef = result.resultComment
        ? buildIssueCommentRef(owner, repo, result.resultComment.id)
        : buildReviewRef(owner, repo, result.pr, result.resultReview.id);
    const resultLink = result.resultComment?.html_url ?? result.resultReview?.html_url ?? null;
    const resultPreview = result.resultComment
        ? extractPreview(result.resultComment.body ?? "")
        : extractPreview(result.resultReview.body ?? "");
    const shouldPrintResultPreview = result.resultComment != null || result.inlineSummaries.length === 0;
    console.log([
        "done",
        `elapsed=${formatElapsed(result.elapsedMs)}`,
        `repo=${result.repo}`,
        `pr=${result.pr}`,
        `result=${resultRef}`,
        resultLink ? `link=${resultLink}` : null,
    ].filter(Boolean).join(" "));
    if (shouldPrintResultPreview) {
        console.log(`preview: ${resultPreview}`);
    }
    if (result.reviewSummaries.length === 0 && result.resultComment) {
        console.log("reviews: none");
    }
    else {
        for (const review of result.reviewSummaries) {
            console.log([
                "review:",
                `ref=${review.ref}`,
                review.link ? `link=${review.link}` : null,
                `state=${review.state}`,
            ].filter(Boolean).join(" "));
            if (review.preview) {
                console.log(`preview: ${review.preview}`);
            }
        }
    }
    if (result.inlineSummaries.length === 0) {
        console.log("inline: none");
    }
    else {
        for (const inline of result.inlineSummaries) {
            console.log([
                "inline:",
                inline.severity ? `severity=${inline.severity}` : null,
                inline.title ? `title=${JSON.stringify(inline.title)}` : null,
                `ref=${inline.ref}`,
                inline.link ? `link=${inline.link}` : null,
                !inline.title && inline.location ? `location=${inline.location}` : null,
            ].filter(Boolean).join(" "));
            if (!inline.title) {
                console.log(`preview: ${inline.preview}`);
            }
        }
    }
}
export async function main(argv = process.argv.slice(2)) {
    const parsedArgs = parseArgs(argv);
    const result = await waitForReviewActivity(parsedArgs);
    printResult(result);
    return result.timedOut ? 2 : 0;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main().then((exitCode) => {
        process.exitCode = exitCode;
    }, (error) => {
        console.error(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
    });
}
