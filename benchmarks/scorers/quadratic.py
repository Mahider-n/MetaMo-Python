"""Defines fitted quadratic equations that calculate emotion scores from six bounded modulator values using learned coefficients, without fuzzy matching."""

import numpy as np

from core.config import (
    M_VALENCE, M_AROUSAL, M_APPROACH, M_RESOLUTION, M_THRESHOLD, M_SECURING, NUM_MODULATORS,
)

EMOTIONS = (
    "happiness", "sadness", "anger", "fear", "excitement", "pride", "love", "hate",
    "gratitude", "hope", "relief", "surprise", "distress", "shame", "guilt", "regret",
    "confusion", "jealousy", "frustration", "irritation", "rage", "depression",
    "anxiety", "horror", "dislike", "contempt",
)
MODULATOR_INDICES = (M_VALENCE, M_AROUSAL, M_APPROACH, M_RESOLUTION, M_THRESHOLD, M_SECURING)

FORMULA_MODEL = "six-modulator-quadratic-v2"
EMOTION_COEFFICIENTS = {
    'happiness': (0.6348052749299843, 0.01710382007190317, 1.5807762856703693, -1.018326046229747, -0.6938221312317399, -0.9003112319030979),
    'sadness': (-2.1132486941149597, 1.174035623014785, 1.4393611790950436, -0.0534717515838628, 0.36563655226001857, -0.7580768841934779),
    'anger': (-2.243169836281673, -0.6864797579180516, 0.4748770972673006, -0.05611007657008921, 0.1915965398905353, 0.20341502275054701),
    'fear': (-0.23133178947538235, -0.009923655862640879, -1.0637399360309687, -0.10901236356324334, 0.39267691556644324, 0.8696514754687619),
    'excitement': (0.1268010022737746, 0.7332119213359374, -0.02868488030767974, -0.08854689129305196, 0.2307884470297959, 0.1989727201650732),
    'pride': (0.966485093386125, -1.5118822201209043, 0.21230180462274856, 0.8437467350438212, -0.13161552396473533, -0.013256671679451534),
    'love': (0.8432740427115678, -0.10540925533894598, 0.4216370213557839, 0.10540925533894598, -0.21081851067789195, -0.21081851067789195),
    'hate': (-0.7921180343813395, 0.39605901719066977, 0.39605901719066977, 0.19802950859533489, 0.09901475429766744, 0.09901475429766744),
    'gratitude': (1.2496907268849213, 0.019495124057249694, 0.23884857267384793, 0.7965284342490472, -0.09925075509468995, -0.2725888028119191),
    'hope': (0.2863098365268434, -0.22327981535568378, -0.7743924166545516, 0.9095652017985536, 1.3253119596301353, 0.8363557790008954),
    'relief': (0.3968392630729245, 0.47917259514104305, -0.4303932793100703, 0.5601648625077427, -0.001584167134182135, 0.17690924802926022),
    'surprise': (-0.03589836981644093, 0.3113820009393721, 0.2393732919696687, -0.31465124174836956, 0.08082001268683589, 0.14584723010801065),
    'distress': (0.9146065062879768, 0.9713791209173249, -2.6499439579442177, -0.7319358532486687, -0.5736877243091886, 0.7145808348246219),
    'shame': (-0.44170901773986226, 0.2969437774853467, 0.010624254708798131, 0.41750371112081724, 0.2799266262752868, 0.06552320742291112),
    'guilt': (-0.7316089268915612, 0.428380907770489, 0.07645862949698382, 0.22019654320698256, 0.27278362029735287, -0.05514368054002297),
    'regret': (-0.6768401385225424, -0.22216812572832148, -0.289126199281981, 0.4623885868825492, 0.23093959315052268, 0.10778829592782496),
    'confusion': (-0.014402060389282267, 0.01865964180628521, 0.01938153749731039, -0.7190698622941847, 0.2767290534043175, 0.18638437737220023),
    'jealousy': (-0.2311900334887288, 0.20470509333786369, 0.30934421009683144, 0.20497505542842917, 0.1341951123654514, 0.09956791804440868),
    'frustration': (-0.38999738178708565, 1.3754703983297423, 1.314577128254867, 0.5140836503320293, 0.3685455793443309, 0.12116154453396688),
    'irritation': (-0.5795261472084394, 0.5208010277075898, 0.33760618886246097, 0.19152377434080375, 0.39799425890089035, 0.22202314505303594),
    'rage': (-0.4755748569061775, 0.4442113235059989, 0.30595891766283423, -0.15165288639686111, -0.2577152083857437, -0.04238655689281545),
    'depression': (-0.6305853614582437, -0.5441191334845717, -0.3505115456378503, 0.1479557340075935, 0.053315654252953745, 0.13167071304615072),
    'anxiety': (0.07247415183337136, 0.04699661288025864, -0.09398963675236771, 0.2651279869232736, 0.17712617645851025, 0.21694257530127087),
    'horror': (-0.19304060462698353, 0.16776567028006334, -0.27148058369517947, 0.15285300785460057, 0.09202693800443273, 0.2915902116979545),
    'dislike': (-0.945138771372702, 0.11919070605985577, -0.07264210397604809, 0.3539653970159674, 0.37038822448306374, 0.03367083470173543),
    'contempt': (-0.804719417926742, 0.6270317761134859, -0.1424930297604396, 0.3411364347198024, 0.534850001385428, 0.2574107734861165),
}


def formula_coefficients():
    """Return fitted weights in emotion order and the six-modulator column order."""
    if set(EMOTION_COEFFICIENTS) != set(EMOTIONS):
        raise ValueError("Expected coefficients for every supported emotion")
    weights = np.array([EMOTION_COEFFICIENTS[name] for name in EMOTIONS], dtype=float)
    if weights.shape != (len(EMOTIONS), 6) or not np.isfinite(weights).all():
        raise ValueError("Expected finite coefficients for every emotion and modulator")
    return weights


def quadratic_features(values):
    """Calculate signed quadratic evidence from six bounded modulator values."""
    values = np.asarray(values, dtype=float)
    if (values.ndim == 0 or values.shape[-1] != 6 or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()):
        raise ValueError("Expected six finite modulators in [0, 1]")
    centered = 2 * values - 1
    return centered * np.abs(centered)


def formula_values(values):
    """Evaluate the current per-emotion quadratic equations."""
    scores = quadratic_features(values) @ formula_coefficients().T
    return {name: scores[..., i] for i, name in enumerate(EMOTIONS)}


def quadratic_feelings(state):
    """Score emotions from all six appraised modulators."""
    if NUM_MODULATORS != 6 or set(MODULATOR_INDICES) != set(range(6)):
        raise ValueError("Quadratic scoring requires the documented six-modulator layout")
    values = state.M.tolist()
    if len(values) != 6 or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Expected six numeric modulator values")
    ordered = [values[index] for index in MODULATOR_INDICES]
    return {name: float(score) for name, score in formula_values(ordered).items()}
