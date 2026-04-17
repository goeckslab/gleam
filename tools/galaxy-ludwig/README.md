# Galaxy-Ludwig

Galaxy-Ludwig is the GLEAM wrapper family that exposes Ludwig configuration, training, evaluation, prediction, visualization, and hyperparameter optimization through Galaxy.

## Included wrappers

- `ludwig_autogenconfig.xml`
- `ludwig_render_config.xml`
- `ludwig_train.xml`
- `ludwig_experiment.xml`
- `ludwig_evaluate.xml`
- `ludwig_predict.xml`
- `ludwig_hyperopt.xml`
- `ludwig_visualize.xml`

## Installation into Galaxy

1. Copy or symlink `tools/galaxy-ludwig` into your Galaxy `tools/` directory.
2. Register the wrappers you want in `tool_conf.xml` or your equivalent tool panel configuration:

   ```xml
   <section id="ludwig" name="Ludwig Applications">
     <tool file="gleam/tools/galaxy-ludwig/ludwig_evaluate.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_experiment.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_hyperopt.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_predict.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_render_config.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_train.xml" />
     <tool file="gleam/tools/galaxy-ludwig/ludwig_visualize.xml" />
   </section>
   ```

3. Configure Galaxy container execution. For example:

   ```yaml
   runners:
     local:
       load: galaxy.jobs.runners.local:LocalJobRunner
       workers: 4
   execution:
     default: local
     environments:
       local:
         runner: local
         docker_enabled: true
   ```

4. If your deployment sanitizes all HTML output, configure Galaxy appropriately so the generated reports can render as intended.

## Container images

- CPU-oriented wrapper image: `quay.io/goeckslab/galaxy-ludwig:0.10.3`
- GPU-oriented wrapper image: `quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1`

Pre-pulling the image can reduce startup latency:

```bash
docker pull quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1
```
