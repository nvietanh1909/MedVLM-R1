from src.rewards.utils import extract_content


class MedicalEntityReward:
    MODALITY_VARIANTS = {
        "computed tomography": ["ct", "computed tomography", "ct scan"],
        "magnetic resonance imaging": ["mri", "magnetic resonance", "mr imaging", "mr scan"],
        "x-ray": ["x-ray", "xray", "radiograph", "plain film"],
        "ultrasound": ["ultrasound", "ultrasonography", "sonography"],
        "digital photography": ["photograph", "clinical photo", "image"],
        "microscopy images": ["microscopy", "histolog", "pathology", "biopsy", "cytolog"],
        "endoscopy": ["endoscop", "colonoscop", "gastroscop"],
        "mammography": ["mammogra", "breast imaging"],
        "fluoroscopy": ["fluoroscop"],
        "angiography": ["angiogra", "arteriogra"],
        "dermoscopy": ["dermoscop", "dermatoscop"],
        "fundus photography": ["fundus", "retinal", "optic disc", "fundoscop"],
        "optical coherence tomography": ["oct", "optical coherence"],
        "pet": ["pet", "positron emission"],
        "ecg": ["ecg", "electrocardiogra", "ekg"],
        "brain imaging": ["brain", "cerebral", "neural", "neuro"],
        "infrared reflectance imaging": ["infrared", "reflectance", "ir imaging"],
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
        "neck": ["neck", "thyroid", "cervical", "laryn", "pharyn"],
        "eye": ["eye", "retina", "optic", "ocular", "cornea", "macula"],
        "skin": ["skin", "dermat", "cutaneous", "melanom", "epiderm"],
        "head": ["head", "skull", "facial", "sinus", "orbit"],
        "cell": ["cell", "cellular", "tissue", "cytolog", "histolog"],
        "extremity": ["extremit", "hand", "foot", "wrist", "ankle", "elbow", "shoulder"],
        "vascular": ["vascular", "artery", "vein", "vessel", "thromb", "embol"],
        "foot": ["foot", "feet", "ankle", "toe", "plantar", "calcaneus"],
        "lower limb": ["lower limb", "leg", "thigh", "knee", "femur", "tibia", "fibula"],
        "upper limb": ["upper limb", "arm", "forearm", "wrist", "humerus", "radius", "ulna"],
        "pelvic cavity": ["pelvi", "bladder", "prostat", "uter", "ovari", "cervi", "sacr"],
        "oral cavity": ["oral", "mouth", "tongue", "teeth", "dental", "gingiv", "palat"],
        "gastrointestinal tract": ["gastrointestin", "stomach", "bowel", "colon", "esophag", "duoden", "rectum", "intestin"],
        "computed tomography": ["ct", "computed tomography"],
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
                variants = self.MODALITY_VARIANTS.get(mod.lower(), [mod.lower()])
                if variants and any(v in text_lower for v in variants):
                    score += 0.5

            if bp:
                variants = self.BODY_PART_VARIANTS.get(bp.lower(), [bp.lower()])
                if variants and any(v in text_lower for v in variants):
                    score += 0.5

            rewards.append(score)
        return rewards
