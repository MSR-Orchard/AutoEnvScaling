# Terminal and image-backed tasks

Use the standard Harbor instruction, environment, reference solution and verifier files. The
assignment, not this overlay, selects the target solver, the difficulty objective, the permitted
images and the resources. Keep the supplied solver evidence in view and say which failure the
task targets.

No particular test framework is required. Choose a custom shell script, Python, pytest or
another installed evaluator that suits the task. All dependencies and public inputs must be
available without downloads at solve time.

If the assignment provides a prebuilt image, use its immutable reference in
`[environment].docker_image` instead of writing a Dockerfile. If the runtime supports setup hooks
or builds generated Dockerfiles, use only the documented route. Some runtimes do not accept
arbitrary Dockerfiles or Compose services; that is a limit of the runtime, not of Harbor's task
format.
