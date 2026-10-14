# Resumen — CI/CD completo del proyecto

Proyecto: **SkinLesionClassifier** · Fecha: 14/10/2026

Cómo quedó armado el flujo de desarrollo completo: repos, CI, imagen Docker y despliegue.

## Repos y registros

| Recurso | Ubicación |
|---|---|
| GitHub | `marlonvalenciamentor-eng/skin-lesion-classifier` |
| GitLab | `gitlab.com/miguel.ortiz01/skin-lesion-classifier` |
| Docker Hub | `miguelortizro/skin-lesion-classifier` |
| Droplet (producción) | `http://167.99.165.18:8501` |

## Flujo completo

```
cambios → git push gitlab main
        → CI de GitLab: test → lint → typecheck → build-image → deploy
        → build-image: docker build + docker push a Docker Hub
        → deploy: SSH al droplet → docker pull + docker run
        → app actualizada en http://167.99.165.18:8501
```

---

## 1. Makefile

Atajos para las tareas de desarrollo:

```bash
make test        # pytest
make lint        # ruff check
make format-check
make typecheck   # mypy --strict
make check       # test + lint + format + typecheck
make run         # streamlit run app.py
make docker      # docker compose up --build
```

---

## 2. Pipeline de GitLab CI (`.gitlab-ci.yml`)

Cinco etapas, todas automáticas en cada push a `main`:

```yaml
image: python:3.13-slim

before_script:
  - apt-get update && apt-get install -y make
  - pip install --no-cache-dir uv
  - uv sync

stages: [test, lint, typecheck, build, deploy]

test:        # make test
lint:        # make lint + make format-check
typecheck:   # make typecheck

build-image: # docker build + docker push a Docker Hub (Docker-in-Docker)
deploy:      # SSH al droplet → docker pull + docker run
```

### Problemas encontrados y solución
- **`make: command not found`** → había que instalar `make` (apt) en el `before_script`.
- **gitlab.com free no da "shared runners"** → los pipelines quedaban en `pending`. Solución: **runner propio** (self-hosted).
- **`Cannot connect to the Docker daemon`** → el runner necesita `privileged = true` (Docker-in-Docker).
- **`HTTP request to an HTTPS server`** → usar Docker sin TLS (`DOCKER_HOST=tcp://docker:2375`, `DOCKER_TLS_CERTDIR=""`).
- **`insufficient scopes`** → el token de Docker Hub debe ser **Read & Write** (no Read-only), y hay que reemplazarlo en la variable.

### Runner (en el droplet, 24/7)

El runner corre en el **droplet** (no en la PC), así el CI funciona sin depender de ninguna máquina local.

```bash
# Instalar en el droplet (Ubuntu):
curl -fsSL https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | bash
apt-get install -y gitlab-runner

# Registrar (el MISMO token sirve para un worker en otra máquina):
gitlab-runner register --non-interactive --url https://gitlab.com --token glrt-XXXX \
  --name droplet-runner --executor docker --docker-image python:3.13-slim

# privado para Docker-in-Docker: privileged = true en /etc/gitlab-runner/config.toml
systemctl enable --now gitlab-runner
```
- Para forzar que use **solo el droplet**: `sudo gitlab-runner stop` en la PC.
- (Antes corría en la laptop, que exigía tener la PC encendida.)

---

## 3. Docker Hub

La imagen se construye y publica sola en cada push.

### Variables CI/CD (GitLab → Settings → CI/CD → Variables)

| Key | Tipo |
|---|---|
| `DOCKERHUB_USER` | Variable (`miguelortizro`) |
| `DOCKERHUB_TOKEN` | Variable **Masked** (Access Token **Read & Write**) |
| `SSH_HOST` | Variable (IP del droplet) |
| `SSH_USER` | Variable (`root`) |
| `SSH_KEY` | **File** (clave privada SSH) |

### Usar la imagen en otra máquina
```bash
docker pull miguelortizro/skin-lesion-classifier:latest
docker run -p 8501:8501 -v skin-lesion-hf:/home/appuser/.cache/huggingface miguelortizro/skin-lesion-classifier:latest
# → http://localhost:8501
```

---

## 4. Deploy por SSH (droplet DigitalOcean)

### El droplet
- **DigitalOcean** droplet Ubuntu, **2 GB RAM** (se hizo resize desde 512 MB, que se colgaba por falta de memoria).
- Docker instalado (`curl -fsSL https://get.docker.com | sh`).
- El contenedor corre con `--restart unless-stopped` (sobrevive reboots).

### Clave SSH
- Clave `ed25519` dedicada: `~/.ssh/gitlab-deploy` (privada) + `.pub` (pública).
- La **pública** se instaló en el droplet al crearlo (DigitalOcean).
- La **privada** se guardó en GitLab como variable `SSH_KEY` (tipo **File**), para que el pipeline entre al droplet sin contraseña.
- ⚠️ **Gotcha:** la clave debe conservar el **newline final**. Si se crea la variable con `$(cat ...)`, el `$()` se come ese newline y OpenSSH falla con *"error in libcrypto: unsupported"*. Solución: subirla leyendo el archivo **crudo** (`--form "value=<archivo"`).

### Job `deploy`
```yaml
deploy:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache openssh-client
    - chmod 400 "$SSH_KEY"
  script:
    - ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$SSH_HOST" \
        "docker pull ...latest && (docker stop ... || true) && (docker rm ... || true) && docker run -d -p 8501:8501 --restart unless-stopped --name skin-lesion-classifier ...latest"
```

---

## 5. MCP servers (opencode)

Se configuraron 3 MCP servers en `~/.config/opencode/opencode.jsonc`:

| MCP | Tipo | Endpoint / comando |
|---|---|---|
| `github` | remote | `https://api.githubcopilot.com/mcp/` (Bearer token) |
| `gitlab` | local | `npx -y @modelcontextprotocol/server-gitlab` |
| `digitalocean` | remote | `https://mcp.digitalocean.com/mcp` (Bearer token) |

Tokens en `~/.bashrc` como variables de entorno:
```bash
export GITHUB_MCP_TOKEN="ghp_..."
export GITLAB_MCP_TOKEN="glpat-..."
export DIGITALOCEAN_API_TOKEN="dop_v1_..."
```
> Los 3 tokens quedaron expuestos durante la sesión → **rotarlos** y actualizar `~/.bashrc`.

---

## 6. Los 4 tipos de token (no mezclar)

| Tipo | Prefijo | Para qué |
|---|---|---|
| GitHub PAT | `ghp_` | `git push` a GitHub + MCP |
| GitLab PAT | `glpat-` | `git push` a GitLab + MCP |
| GitLab Runner registration | `glrt-` | `gitlab-runner register` |
| Docker Hub Access Token | `dckr_pat_` | `docker login` / `docker push` |
| DigitalOcean API | `dop_v1_` | MCP + API |

---

## 7. Estado final

- ✅ Pipeline completo en verde: test → lint → typecheck → build → deploy.
- ✅ Imagen publicada en Docker Hub (build + push automático).
- ✅ Deploy automático por SSH al droplet.
- ✅ App corriendo: **http://167.99.165.18:8501**.
- ✅ Runner en el droplet (24/7, sin depender de la PC).
- ✅ GitHub Actions también ejecuta test + docker build en cada push.
- ✅ MCP servers de GitHub, GitLab y DigitalOcean configurados.
- ✅ Makefile con las tareas de desarrollo.

## 8. Notas de operación

- **El runner corre en el droplet** (24/7) → el CI ya **no necesita la PC**.
- **Cada `git push gitlab main`** reconstruye, publica y redespliega la app.
- **El droplet cuesta ~$12/mes** (por horas, mientras está encendido).
