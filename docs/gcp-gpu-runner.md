# Running GPU workflows on GCP via Cirun

The GPU job defined in `.github/workflows/_gpu.yml` runs entirely inside a container that has all pybind11-cuda-array-interface dependencies pre-installed. This document explains how to build and store:

1. A GCE VM image that already contains Docker and the NVIDIA Container Toolkit so Cirun can boot a runner quickly.
2. A GPU-enabled container image that bundles the compiled test extension, test executables, and Python test dependencies so `_gpu.yaml` only has to execute tests.

## Prerequisites

- `gcloud` CLI authenticated against the target project with permissions to create instances, images and Artifact Registry repositories. Most easily done with a separate GCP project and a service account with the following IAM permissions:
    ```
    Artifact Registry Administrator
    Compute Instance Admin (v1)
    Compute Storage Admin
    Service Account User
    Service Usage Consumer
    ```
- `docker` CLI authenticated against GCP Artifact Registry.
- GPU quota in the selected region/zone (for example, one `nvidia-tesla-t4`).
- Cirun account connected to your GitHub repo.

## Build the GPU host VM image

0. Naming of projects, images, registries are handled by the following environment variables (with their defaults given below, please adjust to your own settings):
    ```
    GCP_ARTIFACT_REGION=us-central1
    GCP_PROJECT=gpu-test-runners
    IMAGE_VERSION (no default, will be picked up by default when building)
    PACKAGE=pybind11-cuda-array-interface-gpu-tests
    REPOSITORY=pybind11-cuda-array-interface
    GCP_ZONE (no default but strongly advised to use a zone in the same region as the artifact store)
    ```

1. Launch and configure an instance that installs Docker CE and the NVIDIA container toolkit by running:

   ```bash
   export GCP_PROJECT="gpu-test-runners"
   export GCP_ZONE="us-central1-b"
   ./scripts/gcp/build_gpu_base_image.sh
   ```

   The helper script provisions a temporary GPU VM based on Ubuntu 22.04, copies `scripts/gcp/install_gpu_host.sh` to it, runs the installer (Docker, NVIDIA driver 580, NVIDIA container toolkit) and saves the stopped disk as a reusable image family (`pybind11-cuda-array-interface-gpu-host` by default). The builder VM defaults to a smaller machine/boot disk and preemptible pricing; override via `MACHINE_TYPE`, `BOOT_DISK_SIZE` or `PREEMPTIBLE`. Set `IMAGE_NAME`, `IMAGE_FAMILY`, `GPU_TYPE`, etc. to override the other defaults.

2. Verify the image is stored in your project:

   ```bash
   gcloud compute images list --filter="name~'pybind11-cuda-array-interface-gpu-host'"
   ```

3. (Optional) Export the image to Cloud Storage if you need to keep an archive or share it across projects:

   ```bash
   gcloud compute images export \
     --image pybind11-cuda-array-interface-gpu-host \
     --destination-uri gs://my-bucket/pybind11-cuda-array-interface-gpu-host.tar.gz
   ```

   You can later re-import the archive into another project with `gcloud compute images import`.

## Use the host image from Cirun

Add a Cirun job that references the image family to `.cirun.yml` (create the file if it does not exist, please adjust to your settings/namings of GCP resources):

```yaml
runners:
  - name: gpu-runner
    cloud: gcp
    gpu: nvidia-tesla-t4
    instance_type: n1-standard-4
    machine_image: projects/gpu-test-runners/global/images/family/pybind11-cuda-array-interface-gpu-host
    preemptible: true
    region: us-central1-b

    labels: self-hosted-cirun-linux-x64-gpu-gcp

    extra_config:
      project_id: gpu-test-runners

      baseDisk:
        diskType: "pd-ssd"
        diskSizeGb: 100

      serviceAccounts:
        - email: gpu-test-runners-sa@gpu-test-runners.iam.gserviceaccount.com
          scopes:
            - "https://www.googleapis.com/auth/cloud-platform"
```

Cirun will now boot runners from your preconfigured image, so the VM already has Docker plus the correct NVIDIA runtime when the GitHub Actions job starts. Adjust the resource labels to match `.github/workflows/_gpu.yml` (`self-hosted-cirun-linux-x64-gpu-gcp`).

## Build and publish the pybind11-cuda-array-interface GPU test container

1. Build the container that contains the precompiled GPU test suite (remember to change environment variables for your settings):

   ```bash
   GCP_PROJECT="gpu-test-runners" GCP_ARTIFACT_REGION="us-central1" PACKAGE="pybind11-cuda-array-interface-gpu-tests" REPOSITORY="pybind11-cuda-array-interface" ./scripts/docker/build_gpu_test_image.sh
   ```

   By default it uses `nvidia/cuda:13.0.3-devel-ubuntu24.04` to compile the tests and the small `nvidia/cuda:13.0.3-base-ubuntu24.04` for the final image. cuDNN and CUDA's general-purpose math libraries are not needed by this test suite. Override these images with `CUDA_IMAGE_DEVEL` and `CUDA_IMAGE_RUNTIME`. The Dockerfile at `docker/gpu-tests.Dockerfile` installs CuPy and pytest, compiles the pytest extension and GoogleTest executable for the T4 runner (`sm_75`), and copies the prepared environment, NVRTC, and CUDA's small header tree into the final stage. These files allow CuPy's runtime-generated kernels to compile without including the multi-gigabyte CUDA development or runtime image. Dependency installation and C++/CUDA compilation therefore happen while publishing the image, not on the Cirun runner.

2. Push to GCP Artifact Registry:

   The command to use for pushing the image will be printed once the `./scripts/docker/build_gpu_test_image.sh` finishes successfully.

   Use the Artifact Registry region that matches your runner zone’s region (for example: zone `us-central1-b` ⇒ host `us-central1-docker.pkg.dev`). Keeping the registry and VM in the same region minimizes egress time/cost.
   For faster pulls/extraction, attach an NVMe local SSD to the runner VM and let the host image place Docker’s data-root on it automatically (see below) (currently not supported by cirun).

## Running the workflow inside the container

The `_gpu.yml` workflow now calls `scripts/docker/run_gpu_tests.sh`, which:

- Pulls the configured image via workflow/repo vars so please set:
    ```
    GCP_ARTIFACT_REGION
    GCP_PROJECT
    REPOSITORY
    PACKAGE
    IMAGE_VERSION
    ```
    as GitHub Actions vars.
- Runs it with `--gpus all`.
- Mounts the checkout at `/workspace` for optional ad-hoc commands. The default test command uses the source and binaries baked into `/opt/pybind11_cuda_array_interface`, so the mount does not hide the precompiled artifacts.
- Runs the baked pytest GPU suite followed by the baked GoogleTest executable.

You can also use the script locally:

```bash
GCP_PROJECT="gpu-test-runners" GCP_ARTIFACT_REGION="us-central1" PACKAGE="pybind11-cuda-array-interface-gpu-tests" REPOSITORY="pybind11-cuda-array-interface" IMAGE_VERSION="Your image version tag" ./scripts/docker/run_gpu_tests.sh
```

Override the command to run ad-hoc checks (for example, `./scripts/docker/run_gpu_tests.sh bash -lc "python -m pytest /opt/pybind11_cuda_array_interface/tests/pytest -k memory"`).

## Summary of storage locations

- **Host VM image** – lives as a Compute Engine image (and optional Cloud Storage export) inside your GCP project. Cirun references it via the image family.
- **GPU test container** – stored in GCP Artifact Registry in the same region as your GPU runners for low-latency pulls. Use `GCP_ZONE`/`GCP_ARTIFACT_REGION` to control the exact location.

With these scripts plus the updated workflow, Cirun can boot a GPU VM that immediately runs the pybind11-cuda-array-interface GPU tox suite inside the prepared container while reusing GitHub Action caches for dependencies and model weights. The host image includes a boot-time service that, if it detects a local SSD (e.g., GCE NVMe local-ssd), will format/mount it at `/mnt/local-ssd` and set Docker’s `data-root` there so image extraction uses NVMe bandwidth. Attach a local SSD to runner VMs (and ensure the service account has Artifact Registry pull access) for the best pull times.
