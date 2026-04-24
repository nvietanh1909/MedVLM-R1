from src.rewards.utils import extract_content
import re

class MedicalEntityReward:
    MODALITY_VARIANTS = {
        "computed tomography": ["ct", "computed tomography", "ct scan"],
        "magnetic resonance imaging": ["mri", "magnetic resonance", "mr imaging", "mr scan"],
        "x-ray": ["x-ray", "xray", "radiograph", "plain film"],
        "ultrasound": ["ultrasound", "ultrasonography", "sonography"],
        "digital photography": ["photograph", "clinical photo"],
        "microscopy images": ["microscopy", "histolog", "pathology", "biopsy", "cytolog"],
        "endoscopy": ["endoscop", "colonoscop", "gastroscop"],
        "mammography": ["mammogra", "breast imaging"],
        "fluoroscopy": ["fluoroscop"],
        "angiography": ["angiogra", "arteriogra"],
        "dermoscopy": ["dermoscop", "dermatoscop"],
        "fundus photography": ["fundus", "retinal", "optic disc", "fundoscop"],
        "infrared reflectance imaging": ["infrared", "reflectance"],
        "optical coherence tomography": ["oct", "optical coherence"],
        "others": [],
    }

    BODY_PART_VARIANTS = {
        "chest": ["chest", "lung", "thorax", "pulmonary", "pleural", "bronch", "mediastin"],
        "brain": ["brain", "cerebral", "intracranial", "cranial", "cortex", "hippocamp"],
        "abdomen": ["abdomen", "abdominal", "liver", "kidney", "bowel", "intestin", "gastric", "hepat", "renal", "spleen", "pancrea"],
        "bone": ["bone", "skeletal", "fracture", "osseous", "joint", "vertebr", "spine", "spinal"],
        "breast": ["breast", "mammograph"],
        "heart": ["heart", "cardiac", "coronary", "myocard", "aort", "valv"],
        "pelvis": ["pelvi", "bladder", "prostat", "uter", "ovari", "cervi"],
        "pelvic cavity": ["pelvi", "bladder", "prostat", "uter", "ovari", "cervi"],
        "neck": ["neck", "thyroid", "cervical", "laryn", "pharyn"],
        "eye": ["eye", "retina", "optic", "ocular", "cornea", "macula"],
        "skin": ["skin", "dermat", "cutaneous", "melanom", "epiderm"],
        "head": ["head", "skull", "facial", "sinus", "orbit"],
        "extremity": ["extremit", "hand", "foot", "wrist", "ankle", "elbow", "shoulder", "limb"],
        "upper limb": ["upper limb", "arm", "hand", "wrist", "elbow", "shoulder", "humer"],
        "lower limb": ["lower limb", "leg", "foot", "ankle", "knee", "femur", "tibia"],
        "foot": ["foot", "ankle", "toe", "plantar", "calcaneus"],
        "oral cavity": ["oral", "mouth", "tongue", "dental", "tooth", "teeth", "gingiv", "palate"],
        "cell": ["cell", "cellular", "cytolog", "histolog"],
        "vascular": ["vascular", "artery", "vein", "vessel", "thromb", "embol"],
        "gastrointestinal tract": ["gastrointestin", "stomach", "bowel", "colon", "esophag", "duoden", "rectum", "intestin"],
        "others": [],
    }

    def __call__(self, completions, modality=None, body_part=None, **kwargs):
        if modality is None and body_part is None:
            return [0.0] * len(completions)

        modalities = modality if modality is not None else [None] * len(completions)
        body_parts = body_part if body_part is not None else [None] * len(completions)

        rewards = []
        for completion, mod, bp in zip(completions, modalities, body_parts):
            score = 0.0
            text_lower = extract_content(completion).lower()

            if mod:
                mod_lower = mod.lower()
                correct_variants = self.MODALITY_VARIANTS.get(mod_lower, [mod_lower])
                if correct_variants and any(re.search(r"\b" + re.escape(v) + r"\b", text_lower) for v in correct_variants):
                    score += 1.0

            if bp:
                bp_lower = bp.lower()
                correct_variants = self.BODY_PART_VARIANTS.get(bp_lower, [bp_lower])
                if correct_variants and any(re.search(r"\b" + re.escape(v) + r"\b", text_lower) for v in correct_variants):
                    score += 1.0

            rewards.append(score)
        return rewards