"""
Wildcards for NAI
Written by anonymous user from arca.live
Modified some code
@URL <TBD>
"""
import os, glob
import random
import re
import os
import fnmatch
import chardet
from typing import Dict, List
from functools import cache

class BracketValidator:
    """
    Validates if all brackets are closed properly with the right type
    """
    TYPES = {
        '{' : '}',
        '[' : ']',
        '(' : ')',
        '<' : '>',
        '__' : '__'
    }
    STARTS = TYPES.keys()
    ENDS = TYPES.values()

    # match [ or { or ( or __
    @staticmethod
    def validate(text):
        queue = []
        for char in text:
            if char in BracketValidator.STARTS:
                queue.append(char)
            elif char in BracketValidator.ENDS:
                if len(queue) == 0:
                    return False
                if char != BracketValidator.TYPES[queue.pop()]:
                    return False
        return len(queue) == 0

    @staticmethod
    def extract_nested_contents(s, processor:callable=None) -> str:
        """
        Recursively extracts contents from a string with nested curly braces.
        Processor is a function that processes the contents of the innermost curly braces. (list -> str)
        """
        # find the most outer curly braces
        if not BracketValidator.validate(s):
            raise ValueError(f"Invalid bracket in {s}")
        # if no curly braces, return the string
        if "|" not in s:
            return s
        string_before_first_bracket = s.split("{", 1)[0]
        string_after_last_bracket = s.rsplit("}", 1)[-1]
        s = s[len(string_before_first_bracket):]
        def helper(subs):
            nested_level = 0
            start = 0
            segments = []

            for i, char in enumerate(subs):
                if char == '{':
                    if nested_level == 0:
                        start = i
                    nested_level += 1
                elif char == '}':
                    nested_level -= 1
                    if nested_level == 0:
                        # Recursively process the nested part
                        helper_result = helper(subs[start + 1:i])
                        if processor is None:
                            # recover as a string {a|b|...}
                            helper_result = "{" + "|".join(helper_result) + "}"
                        else:
                            helper_result = processor(helper_result) # process list to string by some function
                        segments.append(helper_result)
                elif char == '|' and nested_level == 0:
                    segments.append(subs[start:i])
                    start = i + 1

            # Add the last segment
            if start < len(subs):
                segments.append(subs[start:])

            return segments

        # Remove the outermost curly braces and start processing
        groups = helper(s[1:-1])
        if processor is None:
            groups_result = "{" + "|".join(groups) + "}"
        else:
            groups_result = processor(groups)
        return string_before_first_bracket + groups_result + string_after_last_bracket

class wildcards:

    # 가져올 파일 목록
    card_path = os.path.join(os.path.dirname(__file__), "wildcards", "*.txt")
    #card_path=f"{os.getcwd()}\\wildcards\\**\\*.txt"
    print(f"wildcards card_path : ", card_path)

    # 정규식
    #resub  = re.compile(r"(\{)([^\{\}]*)(\})")
    #resub  = re.compile(r"(\{)(((\d+)|(\d+)?-(\d+)?)?\$\$((.*)?\$\$)?)?([^\{\}]*)(\})")
    resub  = re.compile(r"(\{)(((\d+)|(\d+)?-(\d+)?)?\$\$(([^\{\}]*?)\$\$)?)?([^\{\}]*)(\})")
    recard = re.compile(r"(__)(.*?)(__)")

    # 카드 목록
    is_card_Load = False
    cards: Dict[str, List[str]]= {}
    seperator=", "
    loop_max=50

    def process_group(groups: List[str]) -> str:
        """
        Processes a list of strings and returns a string
        """
        # select first
        if isinstance(groups, list):
            chosen = random.choice(groups)
        else:
            chosen = groups
        # if __card__ is in the chosen string, recursively process it
        cards = wildcards.card_loop(chosen)
        return cards
    
    def recursive_process(text:str) -> str:
        """
        Recursively processes the text
        """
        if not wildcards.is_card_Load:
            wildcards.card_load()
        text = wildcards.process_group(text)
        result = BracketValidator.extract_nested_contents(text, wildcards.process_group)
        if '__' in result:
            print(f"Found discrepancy in {result}, check your wildcards")
        return result

    # | 로 입력된것중 하나 가져오기
    def sub(match):
        #print(f"sub : {(match.groups())}")
        try:        
            #m=match.group(2)
            seperator=wildcards.seperator
            s=match.group(3)
            m=match.group(9).split("|")
            p=match.group(8)
            if p:
                seperator=p
                
            if s is None:
                return random.choice(m)
            c=len(m)
            n=int(match.group(4)) if  match.group(4) else None
            if n:

                r=seperator.join(random.sample(m,min(n,c)))
                #print(f"n : {n} ; {r}")
                return r

            n1=match.group(5)
            n2=match.group(6)
            
            if n1 or n2:
                a=min(int(n1 if n1 else c), int(n2 if n2 else c),c)
                b=min(max(int(n1 if n1 else 0), int(n2 if n2 else 0)),c)
                #print(f"ab : {a} ; {b}")
                r=seperator.join(
                    random.sample(
                        m,
                        random.randint(
                            a,b
                        )
                    )
                )
                #n1=int(match.group(5)) if not match.group(5) is None 
                #n2=int(match.group(6)) if not match.group(6) is None 
            else:
                r=seperator.join(
                    random.sample(
                        m,
                        random.randint(
                            0,c
                        )
                    )
                )
            #print(f"12 : {r}")
            return r


        except Exception as e:         
            console.print_exception()
            return ""
            
            

    # | 로 입력된것중 하나 가져오기 반복
    def sub_loop(text):
        """
        selects from {a|b|c} style
        """
        if "|" not in text:
            # final result
            return text
        target_text=text
        for i in range(1, wildcards.loop_max):
            tmp=wildcards.resub.sub(wildcards.sub, target_text)
            #print(f"tmp : {tmp}")
            if target_text==tmp :
                return tmp
            target_text=tmp
        return target_text

    # 카드 중에서 가져오기
    def card(match: re.Match):
        """
        Find the __card__ in the text and replace it with a random card value
        This function is for returning string values
        """
        #print(f"card in  : {match.group(2)}")
        card_lst=fnmatch.filter(wildcards.cards, match.group(2)) # returns list of keys
        if len(card_lst)>0:
            #print(f"card lst : {lst}")
            cd=random.choice(card_lst)
            #print(f"card get : {cd}")
            r=random.choice(wildcards.cards[cd]) # select from the list of values
        else :    
            r= match.group(2)
        #print(f"card out : {r}")
        return r
    def get_card(card_key:str) -> List[str]:
        """
        Returns a list of card values
        """
        # remove __ from both ends
        if card_key.startswith("__") and card_key.endswith("__"):
            card_key = card_key[2:-2]
        if not wildcards.is_card_Load:
            wildcards.card_load()
        if card_key not in wildcards.cards:
            return [card_key]
        return wildcards.cards[card_key]

    # 카드 중에서 가져오기 반복. | 의 것도 처리
    def card_loop(text:str) -> str:
        """
        Main entry point for the application script
        Processes all the wildcards in the text
        """
        target_text=text
        for i in range(1, wildcards.loop_max):
            tmp=wildcards.recard.sub(wildcards.card, target_text) # functional substitution
            #print(f"card deck selected : {tmp}")
            if target_text==tmp :
                # failed to find card
                tmp=wildcards.sub_loop(tmp) # with {a|b|c} style
                
            if target_text==tmp :
                #print(f"card le : {target_text}")
                return tmp
            target_text=tmp
        #print(f"card le : {target_text}")
        return target_text
        
    # 카드 파일 읽기
    def card_load():
        #cards=wildcards.cards
        card_path=wildcards.card_path
        cards = {}
        #print(f"path : {path}")
        files=glob.glob(card_path, recursive=True)
        #print(f"files : {files}")
        
        for file in files:
            basenameAll = os.path.basename(file)
            basename = os.path.relpath(file, os.path.dirname(__file__)).replace("\\", "/").replace("../../wildcards/", "")
            #print(f"basenameAll : {basenameAll}")
            #print(f"basename : {basename}")
            file_nameAll = os.path.splitext(basenameAll)[0]
            file_name = "/"+os.path.splitext(basename)[0]
            #print(f"file_nameAll : {file_nameAll}")
            #print(f"file_name : {file_name}")
            if not file_nameAll in cards:
                cards[file_nameAll]=[]
            if not file_name in cards:
                cards[file_name]=[]
            #print(f"file_name : {file_name}")
            with open(file, "rb") as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"]
            with open(file, "r", encoding=encoding) as f:
                lines = f.readlines()
                for line in lines:
                    line=line.strip()
                    # 주석 빈줄 제외
                    if line.startswith("#") or len(line)==0:
                        continue
                    if not BracketValidator.validate(line):
                        raise ValueError(f"Invalid bracket in {file_nameAll}: {line}")
                    # check if __file_name__ is in the line, recursive error
                    if "__" + file_nameAll + "__" in line:
                        raise ValueError(f"Recursive __{file_nameAll}__ in {file_nameAll}: {line}")
                    cards[file_nameAll]+=[line]
                    cards[file_name]+=[line]
                    #print(f"line : {line}")
            print(f"card file : {file_nameAll} ; {len(cards[file_nameAll])}")
        wildcards.cards=cards
        print(f"cards file count : ", len(wildcards.cards))
        #print(f"cards : {cards.keys()}")
        wildcards.is_card_Load=True

    # 실행기
    def run(text,load=False):
        if text is None or not isinstance(text, str):
            print("text is not str : ",text)
            return None
        if not wildcards.is_card_Load or load:
            wildcards.card_load()

        #print(f"text : {text}")
        #result=wildcards.card_loop(text)
        result = wildcards.recursive_process(text)
        # if __{}__ is still in text, raise error
        if "__" in result:
            print(f"result : {result}")
            raise ValueError("Failed to find card")
        # split by ',', remove empty string/ whitespaces, and join with ', ' again
        result = ", ".join([x for x in result.split(",") if not x.isspace() and len(x) > 0])
        print(f"result : {result}")
        return result

def SD2NAIstyle(wild):
    wild=wild.replace("\(","▤")
    wild=wild.replace("\)","▥")
    wild=wild.replace("(","{")
    wild=wild.replace(")","}")
    wild=wild.replace("▤", "(")
    wild=wild.replace("▥", ")")
    return wild


    

class NAITextWildcards:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "SD2NAI": (["none", "SD2NAI text style"], {"default":"SD2NAI text style"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "refresh": ("INT", {"default": 0, "min": 0, "max": 1}), # 0: no refresh, 1: refresh
        }
        }
    RETURN_TYPES = ("STRING","ASCII")
    FUNCTION = "encode"

    CATEGORY = "NAI"

    def encode(self, seed, text, SD2NAI, refresh):
        random.seed(seed)
        # print(f"original text : ",text)
        r=wildcards.run(text, load=refresh)
        # print(f"wildcard result : ",r)
        if SD2NAI == "SD2NAI text style":
            r=SD2NAIstyle(r)
            # print(f"SD2NAI result : ",r)
        return (r, r)
