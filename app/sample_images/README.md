# Sample images

Images in this folder are served by the sidebar **Use a bundled sample** view
in `app/app.py`.

| File | Source | Purpose |
|------|--------|---------|
| `apple_pie.jpg` | Food-101 validation split | Demo prediction |
| `baklava.jpg` | Food-101 validation split | Demo prediction |
| `bibimbap.jpg` | Food-101 validation split | Demo prediction |

## Known failure case

The capstone problem statement requires at least three sample predictions and
one known failure case. A failure cannot be selected honestly until the final
model has been evaluated, because it depends on the trained weights.

After training, run:

```bash
python src/evaluate.py --model models/best_transfer_model.keras --output evaluation_results
```

The evaluation engine saves high-confidence misclassification images under
`evaluation_results/failures/images/`. The Streamlit app automatically exposes
those images under **Known failure case**. This avoids hard-coding a failure
that may not actually be a failure for the final model.

## Notes

- All bundled images are `.jpg`; the uploader also accepts `.jpeg` and `.png`.
- The evaluation output directory is mounted read-only by Docker Compose.
