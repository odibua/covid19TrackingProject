#Tracking COVID-19 by Ethnicity
**"Never let a good crisis go to waste"**

The purpose of his repository is to enable the collection of real-time data
regarding covid cases/deaths from different counties and states, stratified by race.
In it's most ideal form this repository will:

1. contain real-time data about covid cases/deaths by ethnicity
2. contain this daa for different states/counties

This repository of data contained above has the potential to motivate future health/pandemic 
policies, quantify disparities in health outcomes, and enable research into what factors
exacerbate/alleviate health disparities.

####Packages Used
The main packages used can be found in ```setup.py```. These are

```
autopep8
bokeh
beautifulsoup4
html5lib
numpy>=1.9.0
pandas
pytest
requests
requests_html
scipy
setuptools>=40.0
typing
urllib3>1.25
wheel
PyYAML
```

####Getting Started
1. Install git on your system, following these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
1. Add ssh key to git, following the instructions [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
1. Clone the covid19Tracking repository using ```git clone git@github.com:odibua/covid19TrackingProject.git```
1. Install python on your system by following these [instructions](https://wiki.python.org/moin/BeginnersGuide/Download)
1. Install pip on your system by following these [instructions](https://pip.pypa.io/en/stable/installing/)
1. Navigate to ```covid19Tracking/``` and run ```pip install -e .```
####Overview of Code

####Adding State/County for scraping raw data
1. #####Running scraper locally
2. #####Handling scraping errors

####Processing raw data
1. #####Running processing locally
2. #####Handling processing errors