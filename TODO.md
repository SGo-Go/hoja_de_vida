# Next steps

### Preparation steps
- [ ] prepare local environment:
  - fork this project
  - install [`devcontainer`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension for VS code
  - try [`devcontainer`](https://code.visualstudio.com/docs/devcontainers/containers) from VS code
  - skim through [references](TODO.md#references)

### Coding steps

It is recommended to make the following coding steps as [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for @sgo-go to review.
- [ ] modify [`devcontainer.json`](.devcontainer/devcontainer.json) to add your favorite VS code extensions
  - [git](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)
  - [markdown](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)
  - [docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
  - [latex](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
  - [actions](https://marketplace.visualstudio.com/items?itemName=github.vscode-github-actions)
- [ ] replace direct use of `Dockerfile` by `docker-compose`
- [ ] inject [`FortySecondsCV`](https://github.com/PandaScience/FortySecondsCV) package in Docker image and remove [`FortySecondsCV`](.gitmodules) submodule from repo
- [ ] rewrite `Dockerfile` to produce compact latex docker image that includes only minimum necessary packages
- [ ] modify $\LaTeX$ code of CV from [`devcontainer`]((https://code.visualstudio.com/docs/devcontainers/containers)) using VS code

### Delivery steps

- [ ] prepare release
- [ ] (optional) add [GitHub action](https://habr.com/en/articles/561644/) that publishes your CV in PDF format
- [ ] (optional) use [Docker container action](https://docs.github.com/en/actions/creating-actions/creating-a-docker-container-action)
  ([example](https://habr.com/en/companies/surfstudio/articles/568030/))

# References

- GitHub basics
  - [forks](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
  - [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
  - [releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
  - [actions](https://docs.github.com/en/actions/quickstart)
- GitHub markdown:
  - [basic syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
  - emojies: [finder](https://github-emoji-picker.rickstaa.dev/)
- devcontainers in VS Code
  - Developing inside a Container [MicroSoft](https://code.visualstudio.com/docs/devcontainers/containers)
  - VS Code, python, контейнеры — как обуздать эту триаду и разрабатывать внутри контейнера
    [habr](https://habr.com/en/companies/ruvds/articles/717110/)
  - exapmles:
    - Latex Dev Container
      [:octocat: GitHub](https://github.com/qdm12/latexdevcontainer/tree/master/.devcontainer)
      [DockerHub](https://hub.docker.com/r/qmcgaw/latexdevcontainer)
    - Alternative Latex Dev Container
      [:octocat: GitHub](https://github.com/hegerdes/VSCode-LaTeX-Container/tree/master/.devcontainer)
- $\LaTeX$
  - 40 Seconds CV template for Xe $\LaTeX$
    [GitHub](https://github.com/PandaScience/FortySecondsCV)
- (optional) Actions:
  - Пишем простейший GitHub Action на TypeScript [habr]()
  - Инструкция: как написать собственный GitHub Action на Dart [habr](https://habr.com/en/companies/surfstudio/articles/568030/)

<details><summary>Extra links</summary>

## Extra

- auto-latex: generate and handle latex through github actions
  [post](https://mrturkmen.com/posts/build-release-latex/)
  [GitHub](https://github.com/merkez/latex-on-ci-cd)
- latex-action [GitHub](https://github.com/xu-cheng/latex-action)
- latex-docker [GitHub](https://github.com/xu-cheng/latex-docker)
- GitHub actions for $\LaTeX$
  [post](https://davidegerosa.com/githubforlatex/)
  [GitHub](https://github.com/dgerosa/writeapaper)
- TeX Live docker image [DockerHub](https://hub.docker.com/r/texlive/texlive)
- https://code.visualstudio.com/docs/containers/overview
- https://learn.microsoft.com/en-us/visualstudio/docker/tutorials/docker-tutorial
<!-- https://habr.com/en/companies/akbarsdigital/articles/703554/ -->
<!-- - Xe $\LaTeX$: 40 Seconds CV template [GitHub](https://github.com/PandaScience/FortySecondsCV) -->

<!-- - Пишем расширение для MediaWiki / https://habr.com/en/companies/veeam/articles/544534/
- Почему Google нуждалась в графе знаний / https://habr.com/en/articles/440938/
- Knowledge bases / https://habr.com/en/articles/195650/
- https://developers.google.com/knowledge-graph/reference/rest/v1/ -->

</details>
