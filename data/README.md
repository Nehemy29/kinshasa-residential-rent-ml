# Dataset documentation

`rent_kinshasa_anonymized.csv` contains 150 housing observations from Mombele, Kingabwa, and Limete Résidentiel. It contains no respondent name, telephone number, or exact personal address.

## Modeling fields

The reproducible analysis uses the following predictors:

- `Quartier`: neighborhood;
- `Type_logement`: housing type;
- `Chambres`: number of bedrooms;
- `Surface_m2`: floor area in square meters;
- `Etat`: housing condition;
- `Eau`: water-access mode;
- `Electricite`: electricity access.

The target is `Loyer_raisonnable`, the monthly rent in US dollars perceived as reasonable by the respondent.

## Fields excluded from prediction

Respondent profile, qualitative perception, asking rent, agreed rent, payment periodicity, advance months, guarantee months, and derived price fields are not used as model predictors. Their exclusion prevents direct price leakage and circular prediction.

## Missing values

Fourteen `Loyer_convenu` values are missing. They correspond to housing seekers for whom no rent agreement had occurred. These missing values do not affect model training because `Loyer_convenu` is excluded from the predictors.

## Limitations and sharing condition

The file does not include complete sampling metadata such as interviewer codes, timestamps, exact selection probabilities, or anonymized fine-scale coordinates. Statistical representativeness cannot therefore be established. Public redistribution should occur only after the author verifies that the original consent conditions permit open-data release.

