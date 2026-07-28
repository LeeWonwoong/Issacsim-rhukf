"""probe_mass.py — 런타임 질량/관성 직접 쿼리 (일회성 검증용)

run_sim.py 의 차량 생성부와 동일 경로(ROBOTS['Iris'] + Multirotor + world.reset())를
재현한 뒤, USD 어트리뷰트와 PhysX 런타임 뷰 양쪽에서 질량/관성을 읽어 프린트한다.
PX4 는 띄우지 않는다(px4_autolaunch=False) — 물리 바디 프로퍼티만 보면 되므로.

실행:  ~/isaacsim/python.sh probe_mass.py
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import sys
import numpy as np

_OUT = open("/tmp/probe_mass_out.txt", "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _OUT.write(s + "\n")
    _OUT.flush()


from omni.isaac.core.world import World
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdPhysics
import omni.usd

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

pg = PegasusInterface()
pg._world = World(**pg._world_settings)
world = pg.world
pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

config_multirotor = MultirotorConfig()
config_multirotor.backends = []          # PX4 없이 물리 바디만

vehicle = Multirotor(
    "/World/quadrotor", ROBOTS['Iris'], 0,
    [0.0, 0.0, 0.07],
    Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
    config=config_multirotor)

world.reset()
stage = omni.usd.get_context().get_stage()

log("\n" + "=" * 70)
log("USD asset :", ROBOTS['Iris'])
log("=" * 70)

# ── (1) USD 어트리뷰트 ──
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if not p.startswith("/World/quadrotor"):
        continue
    if not (prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI)):
        continue
    m = UsdPhysics.MassAPI(prim)
    mass = m.GetMassAttr().Get() if m.GetMassAttr() else None
    inert = m.GetDiagonalInertiaAttr().Get() if m.GetDiagonalInertiaAttr() else None
    log(f"[USD] {p:32s} mass={mass}  diagInertia={inert}")

# ── (2) PhysX 런타임 뷰 ──
for path in ["/World/quadrotor/body", "/World/quadrotor"]:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        view = RigidPrimView(prim_paths_expr=path, name="probe_body")
        world.scene.add(view)
        view.initialize()
        log(f"\n[PhysX] path={path}")
        log(f"[PhysX] get_masses()   = {np.asarray(view.get_masses())}")
        try:
            log(f"[PhysX] get_inertias() = {np.asarray(view.get_inertias())}")
        except Exception as e:
            log(f"[PhysX] get_inertias() 실패: {e}")
        try:
            log(f"[PhysX] get_coms()     = {view.get_coms()}")
        except Exception as e:
            log(f"[PhysX] get_coms() 실패: {e}")
        break

# ── (3) UKF 캘리브레이션 값과 대조 ──
from env.ukf_filter import load_calibration
calib = load_calibration()
d = calib['drone']
log("\n[calib] calibration.json drone =", d)

# ── (4) run_sim._apply_body_calib 와 동일한 설정 후 world.reset() 내성 확인 ──
from pxr import Gf
m_t = float(d['mass'])
I_t = (float(d['Ixx']), float(d['Iyy']), float(d['Izz']))
prim = stage.GetPrimAtPath("/World/quadrotor/body")
mass_api = (UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(prim))
mass_api.CreateMassAttr().Set(m_t)
mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*I_t))
view.set_masses(np.array([m_t], dtype=np.float32))
view.set_inertias(np.array([[I_t[0], 0, 0, 0, I_t[1], 0, 0, 0, I_t[2]]], dtype=np.float32))
log(f"\n[set]  직후 mass={np.asarray(view.get_masses()).ravel()[0]:.6f} "
    f"I={np.asarray(view.get_inertias()).ravel()[[0, 4, 8]]}")

world.reset()
log(f"[set]  world.reset() 후 mass={np.asarray(view.get_masses()).ravel()[0]:.6f} "
    f"I={np.asarray(view.get_inertias()).ravel()[[0, 4, 8]]}")
for _ in range(20):
    world.step(render=False)
log(f"[set]  20 step 후 mass={np.asarray(view.get_masses()).ravel()[0]:.6f} "
    f"I={np.asarray(view.get_inertias()).ravel()[[0, 4, 8]]}")

# ── (5) 로터 바디 질량(총 기체질량에 합산되는지) ──
try:
    rv = RigidPrimView(prim_paths_expr="/World/quadrotor/rotor[0-3]", name="probe_rotors")
    world.scene.add(rv)
    rv.initialize()
    rm = np.asarray(rv.get_masses()).ravel()
    ri = np.asarray(rv.get_inertias()).reshape(len(rm), 9)
    rp, _ = rv.get_world_poses()
    bp, _ = view.get_world_poses()
    rp = np.asarray(rp).reshape(len(rm), 3)
    bp = np.asarray(bp).reshape(3)
    log(f"\n[rotors] masses={rm}  sum={rm.sum():.6f}")
    log(f"[rotors] 총 기체질량 = body({m_t:.4f}) + rotors({rm.sum():.4f}) = {m_t + rm.sum():.6f} kg")
    log(f"[rotors] body world pos = {bp}")
    for i in range(len(rm)):
        log(f"[rotors]  r{i}: m={rm[i]:.6f} pos_rel={rp[i]-bp} Idiag={ri[i][[0,4,8]]}")
    # 바디 COM 기준 총 관성(평행축 정리; 로터 자체 관성 + m*d^2)
    d = rp - bp
    par = np.zeros(3)
    for i in range(len(rm)):
        x, y, z = d[i]
        par += rm[i] * np.array([y*y + z*z, x*x + z*z, x*x + y*y])
    own = ri[:, [0, 4, 8]].sum(axis=0)
    log(f"[rotors] 평행축 기여 = {par}")
    log(f"[rotors] 로터 자체관성 합 = {own}")
    log(f"[rotors] 로터 총기여 = {par + own}")
except Exception as e:
    log(f"[rotors] 조회 실패: {e}")

log("=" * 70 + "\n")

simulation_app.close()
