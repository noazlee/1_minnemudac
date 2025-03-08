# 1_minnemudac
## repo for MinneMUDAC 2025

Contributors: Joyce Li, Aiden Guan, Mia Wang, Noah Lee, Anu Jajodia

Stuff we have done so far (on Novice):

in df_cleaning_and_alignment:
* Created Alignment Gender Column, interest columns, key word columns from 'Rationale for Match'
* Converted columns to appropriate data type - numeric, categorical, date
* Investigated missingness - correlated with Match Length - heavily!! missingness NOT random.
* Investigate missingness by date
* Did key word analysis on 'Rationale for Match' - VERY IMPORTANT - added interest columns from that.
* Imputed data with less that 5% missing data and low variance with mean/median/mode
* Tried creating imputing synthetic data in 'Big Days Acceptance to Match' - using RFR pipeline
* Tried ridge regression on whole model (must make a better one later)

Start: (Jupyter notebook guide: https://www.youtube.com/watch?v=5pf0_bpNbkw)
* Git pull
* Create a conda environment with conda create ___
* Add Jupyter to your environment with pip install jupyter
* In your terminal and in your environment, input: juputer notebook

How to contribute:
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
Careful though because the data on the drive is not in csv format. Open the sheet and download it as a csv manually.

