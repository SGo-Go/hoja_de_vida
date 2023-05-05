# CV draft

## Howto modify

1. Cloning 
  ```bash
  git clone https://$USER@github.com/$USER/hoja_de_vida.git
  cd ./hoja_de_vida
  git submodule update --init && cd ./FortySecondsCV && git submodule update --init
  ```

2. Edit and compite
  - with **VS Code**:
    1. press `F1` and type `Open Folder in Container...`
    2. make changes
    3. cd to docs and run `xelatex` to compile
      ```bash
      cd /workspaces/hoja_de_vida/docs && xelatex ./cv.tex
      ```

Testing outside VS code:
```
docker run -it --entrypoint bash texlive/texlive
```
