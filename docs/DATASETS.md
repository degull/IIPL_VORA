# Datasets and checkpoints

VoRA expects paired image restoration data under the directory passed through
`--data-root`. Data is deliberately not included in this repository.

## Expected layout

```text
<data-root>/
├── rain100h/{train,test}/{input,target}/
├── csd/{train,test}/{input,target}/
├── gopro/{train,test}/{input,target}/
├── reside6k/{train,test}/{input,target}/
└── sidd/{train,test}/{input,target}/
```

Each input image needs a target image with the same filename.

## Checkpoints

Keep downloaded backbone weights and training checkpoints in `checkpoints/`.
The directory is ignored by Git. Pass a base model explicitly:

```bash
