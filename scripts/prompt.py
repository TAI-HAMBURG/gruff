import torch
from transformers import GenerationConfig

llama2_chat_family = []

# Updated model families with comprehensive coverage
leo_chat_family = [
    'LeoLM/leo-hessianai-7b-chat',
    'LeoLM/leo-hessianai-7b-chat-bilingual',
    'LeoLM/leo-hessianai-13b-chat',
    'LeoLM/leo-hessianai-70b-chat'
]

llama3_chat_family = []


class RawLanguageModelInstructionTemplate:
    def __init__(self):
        self.instruction_template = ""

    def add_prompt_template(self, text):
        return text


class Llama2ChatInstructionTemplate:
    def __init__(self):
        self.instruction_template = "[INST] {user_message} [/INST]"

    def add_prompt_template(self, text):
        return self.instruction_template.format(user_message=text)


class Llama3ChatInstructionTemplate:
    def __init__(self):
        self.instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def add_prompt_template(self, text):
        return self.instruction_template.format(user_message=text)


class LeoLMChatInstructionTemplate:
    def __init__(self):
        self.instruction_template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def add_prompt_template(self, text):
        return self.instruction_template.format(prompt=text)

def get_pronoun_templates():
    all_templates = []
    base_templates = [
        #"{task}\nGib nur das korrekte Pronomen für die Lücke aus (ohne weitere Erklärung):"
        "{task}\n\nWhat pronoun should be used to fill the blank?",
        "{task}\n\nThe best pronoun to fill in the blank is",
        "Fill in the blank with the correct pronoun.\n\n{task}",
        "What pronoun should be used to fill the blank?\n\n{task}",
    ]
#    german_templates =[
 #       "{task}\n\nWelches Pronomen sollte in die Lücke eingesetzt werden?",
  #      "{task}\n\nDas beste Pronomen, um die Lücke zu füllen, ist",
   #     "Befülle die Lücke mit dem korrekten Pronomen.\n\n{task}",
    #    "Welches Pronomen sollte genutzt werden, um die Lücke zu füllen?\n\n{task}"
    #]

    for t in base_templates:
        all_templates.append(t)
        if 'correct pronoun' in t:
            a = t.replace('correct pronoun', 'appropriate pronoun')
            all_templates.append(a)

    for t in [t for t in all_templates]:
        all_templates.append(t + '\n{options}')

    return all_templates


def get_instruction_template_fns(model_signature):
    if any(model_name in model_signature for model_name in leo_chat_family):
        return LeoLMChatInstructionTemplate()
    elif any(model_name in model_signature for model_name in llama3_chat_family):
        return Llama3ChatInstructionTemplate()
    elif any(model_name in model_signature for model_name in llama2_chat_family):
        return Llama2ChatInstructionTemplate()
    else:
        raise NotImplementedError(f"Instruction template for {model_signature} not implemented")


def prompt_model(sentence, pronoun_type, pronouns, word, article_and_occupation, tokenizer, model, model_type, model_name):
    sentence_with_blank = sentence.replace(pronoun_type, '___')
    instruction_template = get_instruction_template_fns(model_name)
    all_pronoun_templates = get_pronoun_templates()

    options = pronouns
    options_ = 'OPTIONS:\n' + '\n'.join(['- ' + o for o in options])
    gen_config_args = {
        'max_new_tokens': 40,
        'num_beams': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id
    }
    gen_config = GenerationConfig(**gen_config_args)

    for i, pronoun_template in enumerate(all_pronoun_templates):
        filled = pronoun_template.format(task=sentence_with_blank, options=options_)
        filled_with_instruction = instruction_template.add_prompt_template(filled)

        input_ids = tokenizer(filled_with_instruction, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(inputs=input_ids, generation_config=gen_config).cpu().detach()[0]

        input_ids_cpu = input_ids.cpu().detach()[0]

        if 'flan' in model_name:
            decoded_tokens = tokenizer.decode(outputs, skip_special_tokens=True)
        else:
            decoded_tokens = tokenizer.decode(outputs[len(input_ids_cpu):], skip_special_tokens=True)

        decoded_tokens = (decoded_tokens.strip()).replace("\n", " ")

        yield i, decoded_tokens