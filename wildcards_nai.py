#!/usr/bin/env python3
import os
import glob
import random
import re
import fnmatch
import chardet
import logging
import argparse
from tqdm import tqdm
from typing import Dict, List


# ------------------------------------------------------------------------------
# BracketValidator
# ------------------------------------------------------------------------------
class BracketValidator:
    """
    Validates if all brackets are closed properly with the right type.
    Also provides a method to extract nested contents.
    """
    TYPES = {
        '{': '}',
        '[': ']',
        '(': ')',
        '<': '>',
        '__': '__'
    }
    STARTS = TYPES.keys()
    ENDS = TYPES.values()

    @staticmethod
    def validate(text: str) -> bool:
        """
        Checks if brackets {, [, (, <, or __ are properly opened and closed in 'text'.
        Returns True if valid, False otherwise.
        """
        queue = []
        i = 0
        # Because we have a two-char bracket "__", we need a minor tweak:
        while i < len(text):
            # Try matching multi-char bracket: '__'
            if text[i:i+2] == "__":
                # If it's an opening bracket, push it, skip next char
                if len(queue) == 0 or queue[-1] != '__':
                    queue.append('__')
                else:
                    # If we see '__' while last item is also '__', it might be closing
                    top = queue.pop()
                    if BracketValidator.TYPES.get(top) != '__':
                        return False
                i += 2
                continue

            char = text[i]
            if char in BracketValidator.STARTS and char != '_':
                queue.append(char)
            elif char in BracketValidator.ENDS:
                if not queue:
                    return False
                top = queue.pop()
                if BracketValidator.TYPES[top] != char:
                    return False
            i += 1

        return len(queue) == 0

    @staticmethod
    def extract_nested_contents(s: str, processor: callable = None) -> str:
        """
        Recursively processes the innermost curly braces in 's'.
        If 'processor' is given, it is called on the segments extracted from each brace set.
        This function specifically looks for expansions with '|' (like {a|b|c}).
        """
        # First check bracket validity:
        if not BracketValidator.validate(s):
            raise ValueError(f"Invalid bracket structure in: {s}")

        # If there's no pipe, we don't need to do expansions (for this bracket style).
        if "|" not in s:
            return s

        # We'll handle only the outermost curly braces, then delegate to 'processor' for expansions
        def helper(subs: str) -> List[str]:
            """
            A helper that scans 'subs' for nested { ... } constructs. 
            Each group is passed to 'processor' if given, or re-assembled if not.
            """
            nested_level = 0
            start = 0
            segments = []

            i = 0
            while i < len(subs):
                if subs[i] == '{':
                    if nested_level == 0:
                        start = i
                    nested_level += 1
                elif subs[i] == '}':
                    nested_level -= 1
                    if nested_level == 0:
                        # Recurse on the contents found inside these braces
                        segment_content = helper(subs[start + 1:i])
                        # If processor is None, just keep it as { ... } style
                        if processor is None:
                            # Convert list back to a single string with '|'
                            segment_content = "{" + "|".join(segment_content) + "}"
                        else:
                            segment_content = processor(segment_content)
                        segments.append(segment_content)
                elif subs[i] == '|' and nested_level == 0:
                    # The '|' at top-level breaks the text into separate segments
                    segments.append(subs[start:i])
                    start = i + 1
                i += 1

            # Add whatever remains after the last '|' (or if there's no trailing '|' at top-level)
            if start < len(subs):
                segments.append(subs[start:])

            return segments

        # Strip the outer braces if the entire string is enclosed
        # But for safety, we only do that if the string indeed starts with '{' and ends with '}'
        # Attempt to parse from the inside
        s_stripped = s
        if s.startswith("{") and s.endswith("}"):
            s_stripped = s[1:-1]
        groups = helper(s_stripped)

        if processor is None:
            return "{" + "|".join(groups) + "}"
        else:
            return processor(groups)


# ------------------------------------------------------------------------------
# Wildcards
# ------------------------------------------------------------------------------
class Wildcards:
    """
    Main class for managing wildcard expansions and loading wildcard text files.
    References to __filename__ are expanded from the loaded content in wildcards/filename.txt.
    Weighted lines are supported via `$N` at the end of a line in the text file.
    """

    # Path to wildcard .txt files in "wildcards" folder. Use a wildcard pattern to load all .txt.
    card_path = os.path.join(os.path.dirname(__file__), "wildcards", "*.txt")

    # Regex patterns for expansions:
    # This handles e.g. {3$$someSeparator?some|stuff} plus numeric range expansions.
    # Explanation (roughly):
    #  { ( ( (d+)|(d+)?-(d+) )? $$ ( (some text) $$ )? )? ([^{ }]*) }
    resub = re.compile(
        r"(\{)"
        r"(((\d+)|(\d+)?-(\d+)?)?\$\$(([^\{\}]*?)\$\$)?)?"  # optional "n or n-m"? plus $$someSub?$$
        r"([^\{\}]*)"  # the main "a|b|c"
        r"(\})"
    )

    # Matches __something__ expansions
    recard = re.compile(r"(__)(.*?)(__)")

    # Global dictionary of loaded cards (wildcard expansions)
    is_card_load = False
    cards: Dict[str, List[str]] = {}

    # Misc settings
    separator = ", "     # default separator if none specified
    loop_max = 50        # max times to iterate expansions

    @staticmethod
    def process_group(groups: List[str]) -> str:
        """
        Takes a list of candidate expansions 'groups', each possibly weighted by '$N'.
        Returns a single randomly-chosen result (accounting for weighting).
        If the chosen result has further expansions, 'card_loop' is used to expand them.
        """
        if not groups:
            logging.debug("process_group was called with an empty list.")
            return ""

        # Weighted expansions: if an item ends with e.g. "$3", it appears 3 times in 'pool'
        pool = []
        weight_regex = re.compile(r"(.*)\$(\d+)$")
        for g in groups:
            m = weight_regex.match(g)
            if m:
                text_part, w_str = m.group(1), m.group(2)
                weight = int(w_str)
                pool.extend([text_part] * weight)
            else:
                pool.append(g)

        chosen = random.choice(pool)

        # After picking one, expand any further wildcard references with card_loop
        expanded = Wildcards.card_loop(chosen)
        return expanded

    @staticmethod
    def recursive_process(text: str) -> str:
        """
        Orchestrates expansions by first using process_group, then bracket expansions,
        then final checks for leftover underscores.
        """
        if not Wildcards.is_card_load:
            Wildcards.card_load()

        # The main bracket-based expansion
        # 1) We pass 'text' into process_group once. This is a bit unusual; typically you'd
        #    want process_group on bracket sub-groups. But let's keep consistent with the original design.
        text = Wildcards.process_group([text])

        # 2) Then we do bracket expansions with BracketValidator.extract_nested_contents(..., Wildcards.process_group)
        result = BracketValidator.extract_nested_contents(text, Wildcards.process_group)

        # 3) If any leftover __ remain, we warn (or raise).
        #    The original code tried to do expansions again, but let's keep consistent. 
        if '__' in result:
            logging.warning(f"Found potential leftover card reference in '{result}'. Double-check your expansions.")

        return result

    @staticmethod
    def sub(match: re.Match) -> str:
        """
        Regex substitution for bracket expansions {a|b|c}, possibly with numeric/range picks.
        """
        try:
            # Group(3) -> "s", group(9) -> text of 'a|b|c'
            # group(8) -> possible custom separator
            # group(4) -> single numeric count or None
            # group(5) -> first part of range
            # group(6) -> second part of range
            # ...
            custom_separator = Wildcards.separator
            possible_custom_sep = match.group(8)  # might be something like "##" if we had "##"
            if possible_custom_sep:
                custom_separator = possible_custom_sep

            raw_options_str = match.group(9)
            #logging.info(f"sub() expansion: {raw_options_str}")
            options = raw_options_str.split("|") if raw_options_str else []
            count_str = match.group(4)  # single numeric?
            n1 = match.group(5)        # first in range
            n2 = match.group(6)        # second in range

            c = len(options)
            if not options:
                return ""

            # If we have a simple single numeric 'n'
            if count_str and count_str.isdigit():
                n = int(count_str)
                n = min(n, c)  # clamp
                return custom_separator.join(random.sample(options, n))
            rng_count = 1
            # If we have a range 'n1-n2'
            if n1 or n2:
                lower = int(n1) if n1 else 0
                upper = int(n2) if n2 else c
                lower = max(0, min(lower, c))
                upper = max(0, min(upper, c))
                if lower > upper:
                    lower, upper = upper, lower
                rng_count = random.randint(lower, upper)
                return custom_separator.join(random.sample(options, rng_count))
            logging.info(f"Options: {options}, sampling count: {rng_count}")
            # Otherwise, random sample from 0..c

            return custom_separator.join(random.sample(options, rng_count))

        except Exception as e:
            logging.exception("Exception in sub() expansion.")
            return ""

    @staticmethod
    def sub_loop(text: str) -> str:
        """
        Repeatedly applies 'sub' regex expansions on text, up to loop_max times
        or until text no longer changes.
        """
        if "|" not in text:
            return text

        target_text = text
        for _ in range(Wildcards.loop_max):
            tmp = Wildcards.resub.sub(Wildcards.sub, target_text)
            if tmp == target_text:
                return tmp
            target_text = tmp
        return target_text

    @staticmethod
    def card(match: re.Match) -> str:
        """
        Regex substitution that finds '__cardname__' references and replaces them with
        a random line from the loaded card dictionary matching that name (wildcard).
        """
        wildcard_key = match.group(2)  # e.g. "colors" from __colors__
        if not Wildcards.cards:
            logging.warning("No wildcards loaded, returning the matched key unmodified.")
            return wildcard_key

        # This might match multiple wildcard keys (fnmatch)
        possible_keys = fnmatch.filter(Wildcards.cards.keys(), wildcard_key)
        if possible_keys:
            chosen_key = random.choice(possible_keys)
            # pick one random line from that card list
            return random.choice(Wildcards.cards[chosen_key])
        else:
            # fallback: no matching key found
            return wildcard_key

    @staticmethod
    def get_card(card_key: str) -> List[str]:
        """
        Return a list of expansions for the wildcard 'card_key' (no underscores).
        If not found, returns [card_key] as a literal fallback.
        """
        trimmed = card_key.strip("_")
        if not Wildcards.is_card_load:
            Wildcards.card_load()
        if trimmed in Wildcards.cards:
            return Wildcards.cards[trimmed]
        return [card_key]

    @staticmethod
    def card_loop(text: str) -> str:
        """
        Repeatedly apply the wildcard expansions for __somecard__, then
        expand bracket sets with {a|b|c} until stable.
        """
        target_text = text
        for _ in range(Wildcards.loop_max):
            tmp = Wildcards.recard.sub(Wildcards.card, target_text)
            if tmp == target_text:
                # If no changes, try bracket expansions
                tmp = Wildcards.sub_loop(tmp)
                if tmp == target_text:
                    return tmp
            target_text = tmp
        return target_text

    @staticmethod
    def card_load() -> None:
        """
        Loads *.txt files from the 'wildcards/' directory into Wildcards.cards.
        For each line in each file, if it ends with '$N', that line will be added N times.
        """
        # Clear previous data to ensure a fresh load if requested
        Wildcards.cards = {}
        path_pattern = Wildcards.card_path
        files = glob.glob(path_pattern, recursive=True)

        logging.info(f"Loading wildcard files from '{path_pattern}'. Found {len(files)} files.")

        for file in tqdm(files, desc="Loading wildcard files", unit="files"):
            basename_all = os.path.basename(file)
            # e.g. "colors.txt"
            file_name_all = os.path.splitext(basename_all)[0]  # "colors"

            # We'll store expansions under both "colors" and a relative path if desired
            if file_name_all not in Wildcards.cards:
                Wildcards.cards[file_name_all] = []

            try:
                with open(file, "rb") as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)["encoding"] or "utf-8"

                with open(file, "r", encoding=encoding) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        # skip empty or commented lines
                        if not line or line.startswith("#"):
                            continue

                        # Weighted lines? e.g. "blue$3"
                        repeat_match = re.match(r"(.*)\$(\d+)$", line)
                        repeat_count = 1
                        text_part = line
                        if repeat_match:
                            text_part = repeat_match.group(1)
                            repeat_count = int(repeat_match.group(2))

                        # Validate brackets inside the line
                        if not BracketValidator.validate(text_part):
                            raise ValueError(
                                f"Invalid bracket in file '{file}' line: {text_part}"
                            )

                        for _ in range(repeat_count):
                            Wildcards.cards[file_name_all].append(text_part)
            except Exception as e:
                logging.exception(f"Error reading wildcard file {file}")

            logging.debug(f"Loaded {len(Wildcards.cards[file_name_all])} lines into '{file_name_all}'.")

        Wildcards.is_card_load = True
        logging.info(f"Finished loading wildcards. Total distinct keys: {len(Wildcards.cards)}")

    @staticmethod
    def run(text: str, load: bool = False) -> str:
        """
        Main entry point to expand wildcard references in 'text'.
         - load: if True, forces reload of wildcard files.
        """
        if text is None or not isinstance(text, str):
            logging.error("The provided text for run() is not a string.")
            return ""

        if not Wildcards.is_card_load or load:
            Wildcards.card_load()
        prefix_with_comma = "," if text.startswith(",") else ""
        suffix_with_comma = "," if text.endswith(",") else ""
        expanded = Wildcards.recursive_process(text)
        # If any leftover double-underscores remain, it's presumably an unrecognized card reference
        if "__" in expanded:
            logging.error(f"Expansion might be incomplete: leftover __ in: {expanded}")
            raise ValueError(f"Failed to expand all card references in '{expanded}'.")

        # Attempt a final bracket-based expansion if user used leftover { ... }
        final = Wildcards.sub_loop(expanded)
        # Clean up repeated commas or spaces if needed:
        # (Original code did a quick join by comma to remove empties.)
        joined = ", ".join(x for x in final.split(",") if x.strip())
        joined = prefix_with_comma + joined + suffix_with_comma # restore commas if needed
        logging.info(f"Final expanded text: {joined}")
        return joined


# ------------------------------------------------------------------------------
# Utility: Convert from SD parentheses to NAI curly braces
# ------------------------------------------------------------------------------
def SD2NAIstyle(text: str) -> str:
    """
    Convert text from something like (a|b|c) style to {a|b|c} style for NAI usage.
    Escaped parentheses \( \) can be handled as needed.
    """
    text = text.replace("\\(", "▤").replace("\\)", "▥")
    text = text.replace("(", "{").replace(")", "}")
    text = text.replace("▤", "(").replace("▥", ")")
    return text


# ------------------------------------------------------------------------------
# NAITextWildcards
# ------------------------------------------------------------------------------
class NAITextWildcards:
    """
    Example of a class that uses the Wildcards system for expansions,
    then optionally converts parentheses to curly braces for NAI usage.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "sd2nai": (["none", "SD2NAI text style"], {"default":"SD2NAI text style"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "refresh": ("INT", {"default": 0, "min": 0, "max": 1}), # 0: no refresh, 1: refresh
        }
        }
    RETURN_TYPES = ("STRING","ASCII")
    FUNCTION = "encode"

    CATEGORY = "NAI"
    def encode(self, seed: int, text: str, sd2nai: bool, refresh: bool) -> str:
        random.seed(seed)
        expanded = Wildcards.run(text, load=refresh)
        if sd2nai:
            expanded = SD2NAIstyle(expanded)
        return (expanded, expanded.encode("ascii", "replace"))


# ------------------------------------------------------------------------------
# CLI (argparse)
# ------------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Expand wildcards in a given text using bracket expansions and external wildcard files."
    )
    parser.add_argument("--text", type=str, required=True, help="Text with wildcards to expand.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible expansions.")
    parser.add_argument("--refresh", action="store_true",
                        help="If set, forces reloading of wildcard files from disk.")
    parser.add_argument("--sd2nai", action="store_true",
                        help="If set, converts parentheses in the final text to '{' and '}' for NAI usage.")

    args = parser.parse_args()

    # Use our NAITextWildcards class
    nai_expander = NAITextWildcards()
    result = nai_expander.encode(
        seed=args.seed,
        text=args.text,
        sd2nai=args.sd2nai,
        refresh=args.refresh
    )

    print("Expanded Text:")
    print(result)


if __name__ == "__main__":
    main()
