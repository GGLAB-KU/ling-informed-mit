import json
import os.path
from typing import Optional


class Prompter(object):
    """
    A dedicated helper to manage templates and prompt building.
    """

    __slots__ = ("template", "_verbose")

    def __init__(self,
                 template_name: str,
                 template_dir: str,
                 verbose: bool = False):
        self._verbose = verbose
        file_name = os.path.join(template_dir, f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(self,
                        instruction: str,
                        input: Optional[str] = None,
                        label: Optional[str] = None) -> str:
        """
         returns the full prompt from instruction and optional input
         if a label (=response, =output) is provided, it's also appended.
        """
        if input is not None:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)

        if label is not None:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self,
                     output: str,
                     without_response_title: bool = True) -> str:
        response = output.split(self.template["response_split"])[1].strip()
        if without_response_title:
            return response
        else:
            return self.template["response_split"] + response

    def get_input_text(self, output: str) -> str:
        return output.split(self.template["response_split"])[0].strip()
