"""failsafe_params — hover(failsafe) 진입/복귀 시 PX4 rate-loop 파라미터를 MAVLink 로 스위치.
   ★2026-09-11 사용자 승인: "임무 중지 + failsafe 자세 안정화" 를 문자 그대로 구현.
   노브 FAILSAFE_PARAMS="MC_RR_INT_LIM=1.0,MC_PR_INT_LIM=1.0,MC_ROLLRATE_I=0.8,MC_PITCHRATE_I=0.8" (비면 비활성)
   노브 FAILSAFE_MAV_URL (기본 udpin:0.0.0.0:14540 = SITL onboard 링크; 실기는 젯슨↔FC USB serial)
   명목값은 접속 직후 PX4 에서 읽어 저장(param_request_read) → release() 때 원복. sim=실기 같은 코드."""
import os, time, threading
from env.knobs import knob   # ★09-18 env 변수 → YAML 노브

class FailsafeParams:
    def __init__(self, spec=None, url=None, log=print):
        spec = spec if spec is not None else knob('FAILSAFE_PARAMS', '')
        self.enabled = bool(spec.strip()); self.log = log; self.m = None; self.nominal = {}; self.engaged = False
        self.fs = {}
        if not self.enabled: return
        for kv in spec.split(','):
            k, v = kv.split('='); self.fs[k.strip()] = float(v)
        self.url = url or knob('FAILSAFE_MAV_URL', 'udpin:0.0.0.0:14540')
        self._lock = threading.Lock(); self.ready = False
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        from pymavlink import mavutil
        for _ in range(60):   # PX4 부팅 대기 (최대 ~5 분)
            try:
                m = mavutil.mavlink_connection(self.url, source_system=250)
                if m.wait_heartbeat(timeout=5):
                    break
                m.close()
            except Exception as e:
                self.log(f'[FAILSAFE] 접속 재시도: {e}'); time.sleep(3)
        else:
            self.log('[FAILSAFE] ✗ heartbeat 없음 — 비활성'); self.enabled = False; return
        self.m = m
        for k in self.fs:
            v = self._read(k)
            if v is not None: self.nominal[k] = v
        self.ready = True
        self.log(f'[FAILSAFE] 준비: 명목 {self.nominal} → failsafe {self.fs}')

    def _read(self, name, timeout=2.0):
        self.m.mav.param_request_read_send(self.m.target_system, 1, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
            if msg and msg.param_id.strip('\x00') == name: return float(msg.param_value)
        return None

    def _set(self, vals):
        from pymavlink import mavutil
        for k, v in vals.items():
            self.m.mav.param_set_send(self.m.target_system, 1, k.encode(), float(v), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    def _apply(self, vals, tag):
        with self._lock:
            try:
                self._set(vals); time.sleep(0.05); self._set(vals)   # UDP 유실 대비 2회
                self.log(f'[FAILSAFE] {tag}: {vals}')
            except Exception as e:
                self.log(f'[FAILSAFE] ✗ {tag} 실패: {e}')

    def engage(self):
        if not (self.enabled and self.ready) or self.engaged: return
        self.engaged = True; threading.Thread(target=self._apply, args=(self.fs, 'ENGAGE'), daemon=True).start()

    def release(self):
        if not (self.enabled and self.ready) or not self.engaged: return
        self.engaged = False; threading.Thread(target=self._apply, args=(self.nominal, 'RELEASE'), daemon=True).start()
