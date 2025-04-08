# Triple-Intercept-Modeling
Labortatory assignment to find 4th satellite coordinate using triple intercept method.

## How to run?

1. Make sure all necessary libraries are installed. Run this command on terminal:

```sh
pip install -r requirements.txt
```

2. You can change your satelite orbital parameter inside the file `constants.py`

```python
SATELLITE1 = Coordinate(
    x=48853.47967304855, 
    y=-1157502.1971707679,
    z=-229687.1713272969
)

SATELLITE2 = Coordinate(
    x=32862.01144554671,
    y=157502.1971707679,
    z=-208608.9455096204
)

SATELLITE3 = Coordinate(
    x=-167247.2391357287,
    y=-158824.7863056938,
    z=-127309.0503967787
)

# 4th satellite coordinates (the one we are trying to find)
SATELLITE4 = Coordinate(
    x=42122.835,
    y=0.0,
    z=0.0
)
```

3. Run the file `main.py` first. Run this command:

```sh
python main.py
```
