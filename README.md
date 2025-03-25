# Big Brothers Big Sisters Twin Cities Data Analysis

## Overview
This repository contains analysis of Big Brothers Big Sisters Twin Cities data for the MinneMUDAC 2025 challenge. We analyze factors influencing match success and duration between adult mentors (Bigs) and youth mentees (Littles).

## Stuff to look at:
* **Successful matches (Novice)**
* **NLP analysis:**
   * **Temporal analysis - how sentiments change over time**
   * **Words at early vs later stage of match (bring the records right before ending back)**
   * **set of questions that a call coordinator should ask when making Match Support Calls that would be useful in understanding whether or not a match is at-risk for closure in the next 3-6 months. → Close analysis of words right before match ends (seperate of the model)**
   * **Phrase extraction for at risk matches**
* **Length between logs**
* **Consolidating the records**
* **Model prep - survival! - checks for that - multicollinearity? residuals?**

## Key Findings

### Match Length Distribution (novice_questions.rmd)
- Median match length: 16.8 months
- Mean match length: 23.38 months
- Range: 0 to 97.2 months
- Distribution is right-skewed with most matches lasting under 25 months

### Program Type Influence (novice_questions.rmd)
- Community-based programs tend to have longer match durations
- Site-based and Site-based Facilitated programs show similar distributions
- Program type is statistically significant (confirmed by ANOVA tests)

### Time Trends (novice_questions.rmd)
- Older matches (earlier activation dates) show longer durations
- Possible right-censoring effect in the data
- Closure reasons remain relatively consistent over time

### Demographic Factors (novice_questions.rmd)
- **Big's Age:** Positive correlation with match length; age groups 26-35 and 56-65 show better outcomes
- **Big's Gender:** Male Bigs have slightly longer match durations on average
- **Ethnicity Match:** Pairs sharing ethnicity tend to have longer match lengths
- **Geographic Proximity:** Closer proximity correlates with longer matches
- **Occupation:** Arts/Media and Retired Bigs have longer matches; Students have shorter matches
- **Marital Status:** Non-single Bigs tend to have more stable, longer matches

### Interest Alignment (novice_questions.rmd)
- Shared interests, personality compatibility, and commitment level show positive correlations with match length
- Higher number of shared interest categories correlates with longer matches
- Goal alignment is particularly important for successful matches

### Successful Matches (novice_questions.rmd)
- Overall success rate: approximately 36.8%
- Logistic regression model achieved 66.7% accuracy in predicting successful matches
- Cox Proportional Hazards model shows Program Type, Big's Age, and shared interests as key predictors

### Stuff done in undergrad_questions_2.rmd
- Cleaned text into a new column
- Added features on number of logs and proportion of completed logs
- Created topic models with LDA (still need to incorporate topic into df)
- Sentiment analysis on cleaned text
- Added new columns (months till closure, close_to_failure, sentiment columns)
- Output df and df2 for future processing (check slack)

## Contributors
- Joyce Li, Aiden Guan, Mia Wang, Noah Lee

### How to contribute:
* Make sure your code is up to date - git pull
* Make a new branch (eg. issue1_branch) with git checkout -b ___
* Its preferable to code on a file that you know no one else is working on or to make a new file - avoid merge conflicts
* Code on that branch - add it to main when you're ready
* Make sure to leave comments and note which files you are contributing to
* Make a pull request to bring the code back to the main branch when you are done - (also if you are going to entirely code on a new file, you don't really need to make a new branch, just work on main)
* Also: we will be working with github issues, if you are working on something, it should probably be an issue so we can keep track of it. If there is no issue, add a new one. If it is part of another issue, you can edit an existing one.


Warning: Cant push /data folder due to github file constraints. Make sure to download the data on your own and make a Data folder with the following files:
* Novice.csv
* Test_Truncated.csv
* Training.csv
* (Optional) df.csv files - this is to prevent the long wait of waiting for undergrad_questions_2.rmd to load
Careful though because the data on the drive is not in csv format. Open the sheet and download it as a csv manually.

