# News Classifier and Summarizer

## Instruction to run
First make sure your environment has python version 3.10 as this runs only in 3.10
Then run this
```bash
pip install -r requirements.txt
```

To run streamlit locally run this from inside ```app/``` folder
```bash
streamlit run app.py
```
The first run might take quite some time as it needs to download the model for classification and then summarize all 30 articles. Later on these are cached, hence greatly reducing time.