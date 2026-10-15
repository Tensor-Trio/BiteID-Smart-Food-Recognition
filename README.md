# BiteID - Smart Food Recognition

Food image classification across 101 categories using transfer learning on the
[Food-101](https://www.tensorflow.org/datasets/catalog/food101) dataset
(101,000 images). Built with TensorFlow/Keras and deployed through a Streamlit
front end that runs in a single Docker container.

## Responsible use

This system outputs a dish name and a model confidence score. It does **not**
estimate calories or nutrition, and it makes no health claims. The Streamlit
app displays this restriction in its footer.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is required. CPU is fine for the 10-class `SLICE_MODE` subset.
For the full dataset, the pipeline does not cache all resized images in RAM
 because that can require many gigabytes.

## Usage

### Streamlit app

```bash
streamlit run app/app.py
```

Then open http://localhost:8501.
