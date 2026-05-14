from __future__ import annotations

import argparse
import sys

from src.agents.context import RawTranscriptContextProvider
from src.agents.models import QuestionRequest, SummaryRequest
from src.agents.transcript_agent import TranscriptAgent
from src.config import ConfigError, load_settings
from src.observability import cli_run, log_answer, log_summary, log_transcript
from src.transcripts.fetcher import SuperdataTranscriptFetcher
from src.transcripts.storage import ChromaTranscriptStore
from src.transcripts.youtube import extract_video_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yt-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch and cache a transcript")
    fetch.add_argument("url")
    fetch.add_argument("--no-refresh", action="store_true")

    summarize = subparsers.add_parser("summarize", help="Summarize a transcript")
    summarize.add_argument("url")

    ask = subparsers.add_parser("ask", help="Ask a question about a transcript")
    ask.add_argument("url")
    ask.add_argument("question")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(require_keys=True)
        video_id = extract_video_id(args.url)
        with cli_run(args.command, settings, video_id):
            store = ChromaTranscriptStore(settings.chroma_path)
            fetcher = SuperdataTranscriptFetcher(settings.superdata_api_key)
            context_provider = RawTranscriptContextProvider(store, fetcher)
            agent = TranscriptAgent.from_settings(settings, context_provider)

            if args.command == "fetch":
                context = context_provider.get_or_refresh_transcript(
                    video_id, args.url, no_refresh=args.no_refresh
                )
                log_transcript(context.transcript, context.cache_status, settings)
                print(_format_fetch(context.transcript, context.cache_status))
                return 0

            if args.command == "summarize":
                summary = agent.summarize(
                    SummaryRequest(video_id=video_id, source_url=args.url)
                )
                _log_last_context(agent, settings)
                log_summary(summary)
                print(_format_summary(summary.summary, summary.top_findings))
                return 0

            if args.command == "ask":
                answer = agent.answer(
                    QuestionRequest(
                        video_id=video_id,
                        source_url=args.url,
                        question=args.question,
                    )
                )
                _log_last_context(agent, settings)
                log_answer(answer)
                print(answer.answer)
                return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except (ConfigError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _log_last_context(agent: TranscriptAgent, settings) -> None:
    if agent.last_context is None:
        return
    log_transcript(
        agent.last_context.transcript,
        agent.last_context.cache_status,
        settings,
    )


def _format_fetch(transcript, cache_status: str) -> str:
    return "\n".join(
        [
            f"Transcript cached: {transcript.video_id}",
            f"Cache status: {cache_status}",
            f"Characters: {len(transcript.raw_text)}",
        ]
    )


def _format_summary(summary: str, top_findings: list[str]) -> str:
    lines = ["Summary", summary, "", "Top 3 findings"]
    lines.extend(f"{index}. {finding}" for index, finding in enumerate(top_findings, 1))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
