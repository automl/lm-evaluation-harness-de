"""
Take in a YAML, and output all "other" splits with this YAML
"""
import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


SUBJECTS = {
    "abstract_algebra": ("stem", "abstrakte Algebra"),
    "anatomy": ("stem", "Anatomie"),
    "astronomy": ("stem", "Astronomie"),
    "business_ethics": ("other", "Unternehmensethik"),
    "clinical_knowledge": ("other", "klinische Kenntnisse"),
    "college_biology": ("stem", "Biologie in der Hochschule"),
    "college_chemistry": ("stem", "Chemie in der Hochschule"),
    "college_computer_science": ("stem", "Informatik in der Hochschule"),
    "college_mathematics": ("stem", "Mathematik in der Hochschule"),
    "college_medicine": ("other", "Medizin in der Hochschule"),
    "college_physics": ("stem", "Physik in der Hochschule"),
    "computer_security": ("stem", "Computersicherheit"),
    "conceptual_physics": ("stem", "konzeptionelle Physik"),
    "econometrics": ("social_sciences", "Ökonometrie"),
    "electrical_engineering": ("stem", "Elektrotechnik"),
    "elementary_mathematics": ("stem", "Elementarmathematik"),
    "formal_logic": ("humanities", "formale Logik"),
    "global_facts": ("other", "globale Fakten"),
    "high_school_biology": ("stem", "Biologie in der Schule"),
    "high_school_chemistry": ("stem", "Chemie in der Schule"),
    "high_school_computer_science": ("stem", "Informatik in der Schule"),
    "high_school_european_history": ("humanities", "europäische Geschichte in der Schule"),
    "high_school_geography": ("social_sciences", "Geographie in der Schule"),
    "high_school_government_and_politics": ("social_sciences", "Regierung und Politik in der Schule"),
    "high_school_macroeconomics": ("social_sciences", "Makroökonomie in der Schule"),
    "high_school_mathematics": ("stem", "Mathematik in der Schule"),
    "high_school_microeconomics": ("social_sciences", "Mikroökonomie in der Schule"),
    "high_school_physics": ("stem", "Physik in der Schule"),
    "high_school_psychology": ("social_sciences", "Psychologie in der Schule"),
    "high_school_statistics": ("stem", "Statistik in der Schule"),
    "high_school_us_history": ("humanities", "Geschichte der Vereinigten Staaten in der Schule"),
    "high_school_world_history": ("humanities", "Weltgeschichte in der Schule"),
    "human_aging": ("other", "menschliches Altern"),
    "human_sexuality": ("social_sciences", "menschliche Sexualität"),
    "international_law": ("humanities", "internationales Gesetz"),
    "jurisprudence": ("humanities", "Rechtssprechung"),
    "logical_fallacies": ("humanities", "logische Fehlschlüsse"),
    "machine_learning": ("stem", "maschinelles Lernen"),
    "management": ("other", "Management"),
    "marketing": ("other", "Marketing"),
    "medical_genetics": ("other", "medizinische Genetik"),
    "miscellaneous": ("other", "Verschiedenes"),
    "moral_disputes": ("humanities", "moralische Auseinandersetzungen"),
    "moral_scenarios": ("humanities", "moralische Szenarios"),
    "nutrition": ("other", "Ernährung"),
    "philosophy": ("humanities", "Philosophie"),
    "prehistory": ("humanities", "Prähistorie"),
    "professional_accounting": ("other", "professionelle Buchhaltung"),
    "professional_law": ("humanities", "professionelles Recht"),
    "professional_medicine": ("other", "professionelle Medizin"),
    "professional_psychology": ("social_sciences", "professionelle Psychologie"),
    "public_relations": ("social_sciences", "Öffentlichkeitsarbeit"),
    "security_studies": ("social_sciences", "Sicherheitsforschung"),
    "sociology": ("social_sciences", "Soziologie"),
    "us_foreign_policy": ("social_sciences", "US-amerikanische Außenpolitik"),
    "virology": ("other", "Virologie"),
    "world_religions": ("humanities", "Weltreligionen"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="gmmlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, (category, translation) in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            description = f"Im Folgenden sind Multiple-Choice-Fragen (mit Antworten) über {translation}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "group": f"gmmlu_{args.task_prefix}_{category}"
            if args.task_prefix != ""
            else f"gmmlu_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"gmmlu_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"gmmlu_{subject}",
            # "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_subcategories = [
            f"gmmlu_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_subcategories = [f"gmmlu_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"gmmlu_{args.task_prefix}"
                if args.task_prefix != ""
                else "gmmlu",
                "task": mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )

# python _generate_configs.py --base_yaml_path _default_template_yaml --save_prefix_path gmmlu
