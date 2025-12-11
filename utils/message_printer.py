"""Utilities for pretty-printing Claude Agent SDK messages."""

import pprint
from claude_agent_sdk import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)


def print_pretty_message(message: Message) -> None:
    """Pretty print Claude Agent SDK messages with clear formatting.

    Args:
        message: A message from the Claude Agent SDK (SystemMessage, UserMessage,
                AssistantMessage, or ResultMessage)

    Features:
        - Color-coded output for different message types
        - Visual icons for clarity (📝, 💭, 🔧, ✅)
        - Formatted display of text, thinking, and tool use blocks
    """
    # Color codes for terminal output
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    if isinstance(message, SystemMessage):
        print(f"\n{BLUE}{'=' * 80}{RESET}")
        print(f"{BLUE}{BOLD}[SYSTEM MESSAGE]{RESET}")
        print(f"{BLUE}{'=' * 80}{RESET}")
        pprint.pprint(message.data, indent=2, width=100)

    elif isinstance(message, UserMessage):
        print(f"\n{CYAN}{'=' * 80}{RESET}")
        print(f"{CYAN}{BOLD}[USER MESSAGE]{RESET}")
        print(f"{CYAN}{'=' * 80}{RESET}")

        # Handle both string content and block content
        if isinstance(message.content, str):
            print(f"{CYAN}{message.content}{RESET}")
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"{CYAN}{block.text}{RESET}")
                else:
                    # Handle other block types
                    print(f"{CYAN}{block}{RESET}")
        else:
            # Fallback for unexpected content structure
            print(f"{CYAN}{message.content}{RESET}")

    elif isinstance(message, AssistantMessage):
        print(f"\n{GREEN}{'=' * 80}{RESET}")
        print(f"{GREEN}{BOLD}[ASSISTANT MESSAGE]{RESET}")
        print(f"{GREEN}{'=' * 80}{RESET}")

        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"\n{GREEN}📝 Text:{RESET}")
                print(f"{block.text}")

            elif isinstance(block, ThinkingBlock):
                print(f"\n{YELLOW}💭 Thinking:{RESET}")
                print(f"{DIM}{block.thinking}{RESET}")

            elif isinstance(block, ToolUseBlock):
                print(f"\n{MAGENTA}🔧 Tool Use: {BOLD}{block.name}{RESET}")
                print(f"{DIM}Input:{RESET}")
                pprint.pprint(block.input, indent=2, width=100)

    elif isinstance(message, ResultMessage):
        print(f"\n{GREEN}{'=' * 80}{RESET}")
        print(f"{GREEN}{BOLD}[RESULT]{RESET}")
        print(f"{GREEN}{'=' * 80}{RESET}")
        print(f"\n{GREEN}✅ Final Result:{RESET}")
        print(f"{message.result}")

    else:
        # Fallback for unknown message types
        print(f"\n{RED}{'=' * 80}{RESET}")
        print(f"{RED}{BOLD}[UNKNOWN MESSAGE TYPE]{RESET}")
        print(f"{RED}{'=' * 80}{RESET}")
        print(message)


def print_simple_message(message: Message) -> None:
    """Simple message printer without colors (useful for logging or non-terminal output).

    Args:
        message: A message from the Claude Agent SDK
    """
    if isinstance(message, SystemMessage):
        print("\n" + "=" * 80)
        print("[SYSTEM MESSAGE]")
        print("=" * 80)
        pprint.pprint(message.data, indent=2, width=100)

    elif isinstance(message, UserMessage):
        print("\n" + "=" * 80)
        print("[USER MESSAGE]")
        print("=" * 80)

        # Handle both string content and block content
        if isinstance(message.content, str):
            print(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                else:
                    # Handle other block types
                    print(block)
        else:
            # Fallback for unexpected content structure
            print(message.content)

    elif isinstance(message, AssistantMessage):
        print("\n" + "=" * 80)
        print("[ASSISTANT MESSAGE]")
        print("=" * 80)

        for block in message.content:
            if isinstance(block, TextBlock):
                print("\nText:")
                print(block.text)

            elif isinstance(block, ThinkingBlock):
                print("\nThinking:")
                print(block.thinking)

            elif isinstance(block, ToolUseBlock):
                print(f"\nTool Use: {block.name}")
                print("Input:")
                pprint.pprint(block.input, indent=2, width=100)

    elif isinstance(message, ResultMessage):
        print("\n" + "=" * 80)
        print("[RESULT]")
        print("=" * 80)
        print("\nFinal Result:")
        print(message.result)

    else:
        print("\n" + "=" * 80)
        print("[UNKNOWN MESSAGE TYPE]")
        print("=" * 80)
        print(message)
