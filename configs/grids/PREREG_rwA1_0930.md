# 사전 등록: 주 보상 vfinal(1,1.5,1.5,4) vs A-1(1,1,1,3) 판정 — 2026-09-30 01:15, A-1 SWIRL 결과를 열기 전 작성
(사용자 09-30 01:10 위임: "너가 진행해봐 보완할 부분을 너가 보완해서". 근거 = 워크플로 wf_f844bcbd-3a1 리뷰어 §3.)
작성 시점 상태: 원격 n122s_vfinal_rwA1 s42 의 eval.json 이 생성돼 있으나 내용을 읽지 않았다.

## 기본값
주 보상 = vfinal. 동률·미달·유효성 실패는 전부 vfinal.

## 데이터 (surrogate vfinal 무대, 시드 42–46 짝)
- SWIRL A-1: n122s_vfinal_rwA1 (pΔ0.01·R0.25, 중심 설정 하나만; n124s A-1 튜닝 결과는 이 판정에 쓰지 않는다 = 사후 선택 방지, 민감도로만 보고)
- Adam A-1: n122a_vfinal_rwA1 (MSE, lr 3e-4) 과 n125a A-1 lr 1e-3 (MSE)
- 대조 vfinal 짝: SWIRL(n110s a_c s42–44 + n122s_vfinal_cur s45·46) vs Adam(n111a a_pure)

## 지표 (보상 무관, eval.json 에서 계산)
M1 평가 F1 · M2 평가 FPR‰ · M3 공통 운용비용/에피 = (c_fa·FP + c_d·FN − B·탐지사건)/100, 가중 두 개: vfinal(1.5,1.5,4)·(5,5,10) · M4 학습 중 FPR‰ ep61–80.
시드별 DiD_s = (SWIRL−Adam)_{A-1,s} − (SWIRL−Adam)_{vfinal,s}, SWIRL 우세가 양수가 되도록 부호 정렬(M2·M3·M4 는 낮을수록 좋음).

## 규칙 — A-1 을 주 보상으로 채택하려면 아래 전부
- G0 유효성: 5런 완주·평가 crash 0·eval 100에피·원격 GPU 단독(동시 SWIRL 1개) 로그.
- G1 SWIRL 강건성: SWIRL A-1 평균 평가 F1 ≥ 0.873 (vfinal 튜닝 7셀 하한).
- (ii) M1 DiD > 0 가 4/5 시드 이상, 그리고 A-1 의 평균 F1 격차(SWIRL−Adam lr3e-4) > vfinal 의 +0.027.
- (iii) M3 DiD > 0 가 두 가중 모두 4/5 이상.
- (iv) Adam lr 1e-3 @A-1 대비 SWIRL A-1 이 M1·M3(두 가중) 모두 4/5 이상 우세.
- (v) M4 에서 SWIRL 우세 4/5 이상.
어느 쪽이든 A-1 결과는 격차축(G축) 그림에 반드시 포함한다.

## 격차축·모양축 SWIRL 추가 런의 R (결과 보기 전 고정)
원칙 R ≈ TD 잔차 분산. 측정 = Adam(MSE) ep121–200 평균 loss: vfinal 0.45 · A-1 0.28 · A 0.29 · B 0.20 · C 0.10 · (1,5,5,10) 3.5.
→ C: R 0.1 · (1,5,5,10): R 5 (1-2-5 계열 로그 최근접) · 모양축 (1,1,2,4)·(1,2,1,4): c_fa+c_d 가 vfinal 과 같아 R 0.5 · pΔ 는 전부 0.01.
