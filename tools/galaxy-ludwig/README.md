# Galaxy-Ludwig
A library of Galaxy deep learning tools based on Ludwig.

# Install Galaxy-Ludwig into Galaxy
We assume that you have Galaxy running and docker installed in your server/laptop. 
* Create a new folder named ludwig(or whatever) under Galaxy’s tools folder.
* Select and download the branch you want to install and use. Copy all XML, py files under the tools folder in this repo to the ludwig folder(the folder you created in the last step).
* Update `tool_conf.xml` to include Galaxy-Ludwig. See [documentation](https://docs.galaxyproject.org/en/master/admin/tool_panel.html) for more details. This is an example:
```
<section id="ludwig" name="Ludwig Applications">
  <tool file="ludwig/ludwig_evaluate.xml" />
  <tool file="ludwig/ludwig_experiment.xml" />
  <tool file="ludwig/ludwig_hyperopt.xml" />
  <tool file="ludwig/ludwig_predict.xml" />
  <tool file="ludwig/ludwig_render_config.xml" />
  <tool file="ludwig/ludwig_train.xml" />
  <tool file="ludwig/ludwig_visualize.xml" />
</section>
```


* This is an example of a `job_conf.yml` file that you can create to enable Docker for a local Galaxy instance where you want Ludwig-related jobs to run:

```
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
If you are using an older version of Galaxy, then `job_conf.xml` would be something you want to configure instead of `job_conf.yml`. Then you would want to configure destination instead of execution and environment. 
See [documentation](https://docs.galaxyproject.org/en/master/admin/jobs.html#running-jobs-in-containers) for job_conf configuration. 
* If you haven’t set `sanitize_all_html: false` in `galaxy.yml`, please set it to False to enable our HTML report functionality.

# Get Galaxy-Ludwig docker image

This step is optional.
If you want to speed up your runs, execute the following command:
```
docker pull quay.io/goeckslab/galaxy-ludwig-gpu:0.10.1
```

* Should be good to go. 
