# AI Bias Bounty Hackathon `AIDONTKNOW`


#### How to run the project

1. Install the require library. <br>
`
pandas
matplotlib
numpy
seaborn
catboost
scikit-learn
fairlearn
aif360
`
2. Clone the repository. <br>
3. Type `python3 loan_model.py` in your terminal.

#### Demo Video
https://youtu.be/GazB0h52xEw


# 🔍 What we have found
## Class Imbalance

- Criminal record(#1): (no: 9211, yes: 789)
- disability_status(#2):  (no: 8804, yes: 1196)
- Language_Proficiency(#3): (High school: 3014, Some College:1998, Graduate:1540)
- Citizenship_status(#4): (Citizen: 8552, Permanent Resident:991, Visa Holder: 457)
- Employment_Type(#5): (Full time: 6535, Part-time: 1476, Gig:809, Unemployed: 207)


## Bias detection with AIF360

> *__Disclaimer__*
> | DI |  |
> |--|--|
> | 1 | No bias |
> | <1 | Has Bias against topic |
> | >1 | Has Bias against the opposite of topic |  
> 
> | SPD | |
> |--|--|
> | 0 | No bias |
> | <0 | Has Bias against topic |
> | >1 | Has Bias against the opposite of topic |



#### Bias against Black
<img alt="image of bias" src="visualization/Bias1.png" style="width: 500px;" />
Disparate Impact: 0.8203400899931298 <br>
Statistical Parity Difference: -0.07939615454315352

#### Bias against Non-White
<img alt="image of bias" src="visualization/Bias2.png" style="width: 500px;" />
Disparate Impact: 0.8607889732105376 <br>
Statistical Parity Difference: -0.06360423910404034

#### Bias against Non-Native American
<img alt="image of bias" src="visualization/Bias3.png" style="width: 500px;" />
Disparate Impact: 1.0141580860084796 <br>
Statistical Parity Difference: 0.006024717450416883

#### Bias against Non-binary
<img alt="image of bias" src="visualization/Bias4.png" style="width: 500px;" />
Disparate Impact: 0.7727227912835603 <br>
Statistical Parity Difference: -0.0985246815779034

#### Bias against Unemployed
<img alt="image of bias" src="visualization/Bias5.png" style="width: 500px;" />
Disparate Impact: 0.8914381276775055 <br>
Statistical Parity Difference: -0.04948203217842578

#### Bias against Non-White with Criminal Record
<img alt="image of bias" src="visualization/Bias6.png" style="width: 500px;" />
Disparate Impact: 0.7835102201257862 <br>
Statistical Parity Difference: -0.07339418976545842

#### Bias against Non-High Credit Score
<img alt="image of bias" src="visualization/Bias7.png" style="width: 500px;" />
Disparate Impact: 0.6506098087570649 <br>
Statistical Parity Difference: -0.18209597159360852

#### Bias against Non-High Income
<img alt="image of bias" src="visualization/Bias8.png" style="width: 500px;" />
Disparate Impact: 0.6197659385649466 <br>
Statistical Parity Difference: -0.17286558923242212

#### Bias against Non-High Income with Criminal Record
<img alt="image of bias" src="visualization/Bias9.png" style="width: 500px;" />
Disparate Impact: 0.4472864404000656 <br>
Statistical Parity Difference: -0.18477814015950886

---
After that we decide to use fairlearn model to evaluate fairness the sensitive cases ('Gender' and 'Race') <br>
*Before Mitigation*
<img alt="graph of gender and race" src="visualization/beforeMitigation.png" />
*After Mitigation with ThresholdOptimizer from fairlearn* 
<img alt="graph of gender and race" src="visualization/afterMitigation.png" />

