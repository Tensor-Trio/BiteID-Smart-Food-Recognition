# Sample images

Images in this folder are served by the sidebar **Use a bundled sample** view
in `app/app.py`.

## Known failure case

The capstone problem statement requires at least three sample predictions and
one known failure case.

The evaluation engine saves high-confidence misclassification images under
`evaluation_results/failures/images/`. The Streamlit app automatically exposes
those images under **Known failure case**. This avoids hard-coding a failure
that may not actually be a failure for the final model.

## Notes

- All bundled images are `.jpg`; the uploader also accepts `.jpeg` and `.png`.
- The evaluation output directory is mounted read-only by Docker Compose.
