import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import RAPIER from "https://cdn.jsdelivr.net/npm/@dimforge/rapier3d-compat@0.17.3/+esm";
import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs";

const video = document.querySelector(".camera-feed");
const trackingCanvas = document.querySelector(".tracking-canvas");
const trackingCtx = trackingCanvas.getContext("2d");
const threeCanvas = document.querySelector(".three-scene");
const stage = document.querySelector(".stage");
const cursor = document.querySelector(".gesture-cursor");
const toast = document.querySelector(".toast");
const statusText = document.querySelector(".status-text");
const statusDot = document.querySelector(".status-dot");
const buttons = [...document.querySelectorAll(".gesture-button")];

const MODEL_URL = "./assets/models/escaparate.glb?v=2";
const STATIC_SCENE_NAMES = new Set(["cube", "cube001"]);
const HEAD_CAMERA_SCALE = {
  x: 0.42,
  y: 0.26,
};
const HEAD_CAMERA_ROTATION = {
  yaw: 0.28,
  pitch: 0.18,
};
const HEAD_TURN_ROTATION = {
  yaw: 0.32,
  pitch: 0.2,
};
const SHOWCASE_DIRECTION = -1;
const STATIC_SURFACE_HALF_THICKNESS = 0.04;

const state = {
  rapier: null,
  physicsWorld: null,
  renderer: null,
  scene: null,
  camera: null,
  cameraBasePosition: null,
  cameraBaseQuaternion: null,
  cameraTarget: new THREE.Vector3(),
  cameraBaseOffset: new THREE.Vector3(),
  faceLandmarker: null,
  handLandmarker: null,
  raycaster: new THREE.Raycaster(),
  pointerNdc: new THREE.Vector2(),
  pointerRayDirection: new THREE.Vector3(),
  grabPlane: new THREE.Plane(),
  grabOffset: new THREE.Vector3(),
  hoverHitPoint: new THREE.Vector3(),
  hoverBox: null,
  hoveredMesh: null,
  interactiveMeshes: [],
  meshBodies: new Map(),
  modelRoot: null,
  grabTargetPosition: new THREE.Vector3(),
  grabVelocity: new THREE.Vector3(),
  lastGrabTime: 0,
  lastVideoTime: -1,
  parallaxX: 0,
  parallaxY: 0,
  headTurnX: 0,
  headTurnY: 0,
  grabbedMesh: null,
  grabbedBody: null,
  pinchActive: false,
};

const setStatus = (message, ready = false) => {
  statusText.textContent = message;
  statusDot.classList.toggle("is-ready", ready);
};

let toastTimeout;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(toastTimeout);
  toastTimeout = window.setTimeout(() => {
    toast.classList.remove("is-visible");
  }, 1800);
}

const lerp = (from, to, amount) => from + (to - from) * amount;

function isStaticSceneMesh(mesh) {
  const normalizeName = (value) => value.toLowerCase().replace(/[^a-z0-9]/g, "");
  const meshNames = [mesh.name, mesh.parent?.name].filter(Boolean).map(normalizeName);
  return meshNames.some((name) => STATIC_SCENE_NAMES.has(name));
}

function toRapierVector(vector) {
  return { x: vector.x, y: vector.y, z: vector.z };
}

function toRapierQuaternion(quaternion) {
  return { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w };
}

async function setupPhysics() {
  await RAPIER.init();
  state.rapier = RAPIER;
  state.physicsWorld = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
}

function createRigidBodyForMesh(mesh) {
  if (!state.physicsWorld || !mesh.geometry) return;

  mesh.geometry.computeBoundingBox();
  if (!mesh.geometry.boundingBox) return;

  const worldPosition = new THREE.Vector3();
  const worldQuaternion = new THREE.Quaternion();
  const worldScale = new THREE.Vector3();
  mesh.getWorldPosition(worldPosition);
  mesh.getWorldQuaternion(worldQuaternion);
  mesh.getWorldScale(worldScale);

  const localBox = mesh.geometry.boundingBox.clone();
  const localSize = localBox.getSize(new THREE.Vector3());
  const localCenter = localBox.getCenter(new THREE.Vector3());
  const halfExtents = new THREE.Vector3(
    Math.max((localSize.x * worldScale.x) / 2, 0.05),
    Math.max((localSize.y * worldScale.y) / 2, 0.05),
    Math.max((localSize.z * worldScale.z) / 2, 0.05),
  );

  const bodyDesc = isStaticSceneMesh(mesh)
    ? state.rapier.RigidBodyDesc.fixed()
    : state.rapier.RigidBodyDesc.dynamic();

  bodyDesc.setTranslation(worldPosition.x, worldPosition.y, worldPosition.z);
  bodyDesc.setRotation(toRapierQuaternion(worldQuaternion));

  if (!isStaticSceneMesh(mesh)) {
    bodyDesc.setLinearDamping(2.8);
    bodyDesc.setAngularDamping(4.2);
  }

  const body = state.physicsWorld.createRigidBody(bodyDesc);
  if (isStaticSceneMesh(mesh)) {
    const baseX = localCenter.x * worldScale.x;
    const baseY = localCenter.y * worldScale.y;
    const baseZ = localCenter.z * worldScale.z;
    const colliders = [
      {
        desc: state.rapier.ColliderDesc.cuboid(
          halfExtents.x,
          STATIC_SURFACE_HALF_THICKNESS,
          halfExtents.z,
        ),
        translation: {
          x: baseX,
          y: baseY - halfExtents.y + STATIC_SURFACE_HALF_THICKNESS,
          z: baseZ,
        },
      },
      {
        desc: state.rapier.ColliderDesc.cuboid(
          STATIC_SURFACE_HALF_THICKNESS,
          halfExtents.y,
          halfExtents.z,
        ),
        translation: {
          x: baseX - halfExtents.x + STATIC_SURFACE_HALF_THICKNESS,
          y: baseY,
          z: baseZ,
        },
      },
      {
        desc: state.rapier.ColliderDesc.cuboid(
          STATIC_SURFACE_HALF_THICKNESS,
          halfExtents.y,
          halfExtents.z,
        ),
        translation: {
          x: baseX + halfExtents.x - STATIC_SURFACE_HALF_THICKNESS,
          y: baseY,
          z: baseZ,
        },
      },
    ];

    colliders.forEach(({ desc, translation }) => {
      desc.setTranslation(translation.x, translation.y, translation.z);
      desc.setFriction(1.1);
      desc.setRestitution(0.05);
      state.physicsWorld.createCollider(desc, body);
    });
  } else {
    const colliderDesc = state.rapier.ColliderDesc.cuboid(
      halfExtents.x,
      halfExtents.y,
      halfExtents.z,
    );
    colliderDesc.setTranslation(
      localCenter.x * worldScale.x,
      localCenter.y * worldScale.y,
      localCenter.z * worldScale.z,
    );
    colliderDesc.setFriction(1.1);
    colliderDesc.setRestitution(0.05);
    state.physicsWorld.createCollider(colliderDesc, body);
  }

  mesh.userData.physicsBody = body;
  state.meshBodies.set(mesh, body);
}

function syncMeshesFromPhysics() {
  state.meshBodies.forEach((body, mesh) => {
    const translation = body.translation();
    const rotation = body.rotation();
    mesh.position.set(translation.x, translation.y, translation.z);
    mesh.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
  });
}

async function setupThreeScene() {
  const renderer = new THREE.WebGLRenderer({
    canvas: threeCanvas,
    antialias: true,
    alpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.65));

  const keyLight = new THREE.DirectionalLight(0xffffff, 2);
  keyLight.position.set(3, 4, 5);
  scene.add(keyLight);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(MODEL_URL);
  const modelRoot = gltf.scene;
  scene.add(modelRoot);
  modelRoot.updateMatrixWorld(true);

  modelRoot.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
      child.material = Array.isArray(child.material)
        ? child.material.map((material) => material.clone())
        : child.material.clone();
      child.userData.hoverMaterials = [];
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        child.userData.hoverMaterials.push({
          material,
          emissive: material.emissive?.clone(),
          emissiveIntensity: material.emissiveIntensity ?? 0,
        });
      });
      if (!isStaticSceneMesh(child)) {
        state.interactiveMeshes.push(child);
      }
    }
  });

  const blenderCameras = [];
  modelRoot.traverse((child) => {
    if (child.isCamera) {
      blenderCameras.push(child);
    }
  });

  const blenderCamera = blenderCameras[0] ?? gltf.cameras.find((item) => item.isCamera);
  if (!blenderCamera) {
    throw new Error("El GLB no trae camara de Blender");
  }

  scene.attach(blenderCamera);
  blenderCameras
    .filter((camera) => camera !== blenderCamera)
    .forEach((camera) => camera.parent?.remove(camera));

  const hoverBox = new THREE.BoxHelper(modelRoot, 0x38d6bd);
  hoverBox.visible = false;
  scene.add(hoverBox);

  const modelBox = new THREE.Box3().setFromObject(modelRoot);
  modelBox.getCenter(state.cameraTarget);

  blenderCamera.updateProjectionMatrix();

  state.renderer = renderer;
  state.scene = scene;
  state.camera = blenderCamera;
  state.cameraBasePosition = blenderCamera.position.clone();
  state.cameraBaseQuaternion = blenderCamera.quaternion.clone();
  state.cameraBaseOffset.copy(state.cameraBasePosition).sub(state.cameraTarget);
  state.hoverBox = hoverBox;
  state.modelRoot = modelRoot;

  modelRoot.traverse((child) => {
    if (child.isMesh) {
      createRigidBodyForMesh(child);
    }
  });
  syncMeshesFromPhysics();
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
}

async function setupTracking() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm",
  );

  state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });

  state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

function resizeTrackingCanvas() {
  const rect = video.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  trackingCanvas.width = Math.round(rect.width * scale);
  trackingCanvas.height = Math.round(rect.height * scale);
  trackingCtx.setTransform(scale, 0, 0, scale, 0, 0);
}

function drawHandPreview(hand) {
  const rect = trackingCanvas.getBoundingClientRect();
  trackingCtx.clearRect(0, 0, rect.width, rect.height);

  if (!hand) return;

  trackingCtx.fillStyle = "rgba(58, 214, 189, 0.92)";
  for (const point of hand) {
    trackingCtx.beginPath();
    trackingCtx.arc(point.x * rect.width, point.y * rect.height, 3, 0, Math.PI * 2);
    trackingCtx.fill();
  }
}

function updateParallax(face) {
  if (!face) {
    state.parallaxX = lerp(state.parallaxX, 0, 0.08);
    state.parallaxY = lerp(state.parallaxY, 0, 0.08);
    state.headTurnX = lerp(state.headTurnX, 0, 0.08);
    state.headTurnY = lerp(state.headTurnY, 0, 0.08);
    return;
  }

  const nose = face[1];
  const leftEye = face[33];
  const rightEye = face[263];
  const forehead = face[10];
  const chin = face[152];
  const eyeCenter = {
    x: (leftEye.x + rightEye.x) / 2,
    y: (leftEye.y + rightEye.y) / 2,
  };
  const verticalCenter = (forehead.y + chin.y) / 2;

  state.parallaxX = lerp(state.parallaxX, nose.x - 0.5, 0.16);
  state.parallaxY = lerp(state.parallaxY, nose.y - 0.5, 0.16);
  state.headTurnX = lerp(state.headTurnX, (nose.x - eyeCenter.x) * 4, 0.16);
  state.headTurnY = lerp(state.headTurnY, (nose.y - verticalCenter) * 3, 0.16);
}

function resetMeshHover(mesh) {
  if (!mesh?.userData.hoverMaterials) return;

  mesh.userData.hoverMaterials.forEach(({ material, emissive, emissiveIntensity }) => {
    if (material.emissive && emissive) {
      material.emissive.copy(emissive);
      material.emissiveIntensity = emissiveIntensity;
    }
  });
}

function applyMeshHover(mesh) {
  if (!mesh?.userData.hoverMaterials) return;

  mesh.userData.hoverMaterials.forEach(({ material }) => {
    if (material.emissive) {
      material.emissive.set("#38d6bd");
      material.emissiveIntensity = 0.55;
    }
  });
}

function setHoveredMesh(mesh) {
  if (state.hoveredMesh === mesh) return;

  resetMeshHover(state.hoveredMesh);
  state.hoveredMesh = mesh;
  applyMeshHover(state.hoveredMesh);

  if (state.hoverBox) {
    state.hoverBox.visible = Boolean(mesh);
    if (mesh) {
      state.hoverBox.setFromObject(mesh);
    }
  }
}

function getPointerRay(screenX, screenY) {
  const rect = stage.getBoundingClientRect();
  state.pointerNdc.x = ((screenX - rect.left) / rect.width) * 2 - 1;
  state.pointerNdc.y = -(((screenY - rect.top) / rect.height) * 2 - 1);
  state.raycaster.setFromCamera(state.pointerNdc, state.camera);
  return state.raycaster.ray;
}

function applyBlenderCameraTransform(width, height) {
  if (!state.camera) return;

  const yawAmount =
    state.parallaxX * HEAD_CAMERA_ROTATION.yaw * SHOWCASE_DIRECTION +
    state.headTurnX * HEAD_TURN_ROTATION.yaw * -SHOWCASE_DIRECTION;
  const pitchAmount =
    state.parallaxY * HEAD_CAMERA_ROTATION.pitch * -SHOWCASE_DIRECTION +
    state.headTurnY * HEAD_TURN_ROTATION.pitch * SHOWCASE_DIRECTION;
  const yawQuaternion = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 1, 0),
    yawAmount,
  );
  const rightAxis = new THREE.Vector3(1, 0, 0).applyQuaternion(state.cameraBaseQuaternion);
  const pitchQuaternion = new THREE.Quaternion().setFromAxisAngle(rightAxis, pitchAmount);
  const orbitOffset = state.cameraBaseOffset
    .clone()
    .applyQuaternion(yawQuaternion)
    .applyQuaternion(pitchQuaternion);
  const localShift = new THREE.Vector3(
    state.parallaxX * HEAD_CAMERA_SCALE.x * SHOWCASE_DIRECTION,
    state.parallaxY * HEAD_CAMERA_SCALE.y * -SHOWCASE_DIRECTION,
    0,
  ).applyQuaternion(state.cameraBaseQuaternion);

  state.camera.aspect = width / height;
  state.camera.position.copy(state.cameraTarget).add(orbitOffset).add(localShift);
  state.camera.up.set(0, 1, 0);
  state.camera.lookAt(state.cameraTarget);
  state.camera.updateProjectionMatrix();
}

function updateSceneHover(screenX, screenY, htmlTarget) {
  if (!state.camera || !state.interactiveMeshes.length || htmlTarget) {
    setHoveredMesh(null);
    return;
  }

  const rect = stage.getBoundingClientRect();
  if (
    screenX < rect.left ||
    screenX > rect.right ||
    screenY < rect.top ||
    screenY > rect.bottom
  ) {
    setHoveredMesh(null);
    return;
  }

  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  applyBlenderCameraTransform(width, height);

  const ray = getPointerRay(screenX, screenY);
  const [hit] = state.raycaster.intersectObjects(state.interactiveMeshes, false);
  setHoveredMesh(hit?.object ?? null);
  if (hit) {
    state.hoverHitPoint.copy(hit.point);
  } else {
    state.hoverHitPoint.set(0, 0, 0);
  }
  return ray;
}

function startGrab(screenX, screenY) {
  if (!state.hoveredMesh) return;

  const body = state.meshBodies.get(state.hoveredMesh);
  if (!body || body.isFixed()) return;

  const objectPosition = new THREE.Vector3();
  state.hoveredMesh.getWorldPosition(objectPosition);
  state.camera.getWorldDirection(state.pointerRayDirection);
  state.grabPlane.setFromNormalAndCoplanarPoint(state.pointerRayDirection, objectPosition);

  const ray = getPointerRay(screenX, screenY);
  const hitOnPlane = new THREE.Vector3();
  if (!ray.intersectPlane(state.grabPlane, hitOnPlane)) return;

  state.grabOffset.copy(objectPosition).sub(hitOnPlane);
  state.grabTargetPosition.copy(objectPosition);
  state.grabVelocity.set(0, 0, 0);
  state.lastGrabTime = performance.now();
  state.grabbedMesh = state.hoveredMesh;
  state.grabbedBody = body;
  body.setLinvel({ x: 0, y: 0, z: 0 }, true);
  body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  body.setBodyType(state.rapier.RigidBodyType.KinematicPositionBased, true);
}

function updateGrab(screenX, screenY) {
  if (!state.grabbedBody || !state.grabbedMesh) return;

  const ray = getPointerRay(screenX, screenY);
  const hitOnPlane = new THREE.Vector3();
  if (!ray.intersectPlane(state.grabPlane, hitOnPlane)) return;

  const targetPosition = hitOnPlane.add(state.grabOffset);
  const now = performance.now();
  const deltaSeconds = Math.max((now - state.lastGrabTime) / 1000, 1 / 240);
  state.grabVelocity
    .copy(targetPosition)
    .sub(state.grabTargetPosition)
    .divideScalar(deltaSeconds);
  state.grabTargetPosition.copy(targetPosition);
  state.lastGrabTime = now;

  state.grabbedBody.setNextKinematicTranslation(toRapierVector(targetPosition));
  state.grabbedBody.setNextKinematicRotation(toRapierQuaternion(state.grabbedMesh.quaternion));
}

function releaseGrab() {
  if (!state.grabbedBody) return;

  state.grabbedBody.setBodyType(state.rapier.RigidBodyType.Dynamic, true);
  state.grabbedBody.setLinvel(toRapierVector(state.grabVelocity), true);
  state.grabbedBody.wakeUp();
  state.grabVelocity.set(0, 0, 0);
  state.grabbedMesh = null;
  state.grabbedBody = null;
}

function updateHandInteraction(hand) {
  if (!hand) {
    releaseGrab();
    cursor.classList.remove("is-visible", "is-pinching");
    buttons.forEach((button) => button.classList.remove("is-targeted", "is-pressed"));
    setHoveredMesh(null);
    state.pinchActive = false;
    return;
  }

  const indexTip = hand[8];
  const thumbTip = hand[4];
  const pointerX = (indexTip.x + thumbTip.x) / 2;
  const pointerY = (indexTip.y + thumbTip.y) / 2;
  const screenX = (1 - pointerX) * window.innerWidth;
  const screenY = pointerY * window.innerHeight;
  const pinchDistance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
  const isPinching = pinchDistance < 0.055;

  cursor.style.left = `${screenX}px`;
  cursor.style.top = `${screenY}px`;
  cursor.classList.add("is-visible");
  cursor.classList.toggle("is-pinching", isPinching);

  const target = document.elementFromPoint(screenX, screenY);
  const button = target?.closest?.(".gesture-button") ?? null;
  updateSceneHover(screenX, screenY, button);

  buttons.forEach((item) => {
    item.classList.toggle("is-targeted", item === button);
    item.classList.toggle("is-pressed", item === button && isPinching);
  });

  if (state.grabbedBody && isPinching) {
    updateGrab(screenX, screenY);
  } else if (button && isPinching && !state.pinchActive) {
    button.click();
  } else if (state.hoveredMesh && isPinching && !state.pinchActive) {
    startGrab(screenX, screenY);
  }

  if (!isPinching && state.pinchActive) {
    releaseGrab();
  }

  state.pinchActive = isPinching;
}

function renderScene() {
  if (!state.renderer || !state.scene || !state.camera || !state.physicsWorld) return;

  const rect = stage.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  state.renderer.setSize(width, height, false);

  state.physicsWorld.step();
  syncMeshesFromPhysics();

  applyBlenderCameraTransform(width, height);

  if (state.hoverBox?.visible) {
    state.hoverBox.update();
  }

  state.renderer.render(state.scene, state.camera);
}

function tick() {
  resizeTrackingCanvas();

  if (
    state.faceLandmarker &&
    state.handLandmarker &&
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = video.currentTime;
    const now = performance.now();
    const faceResult = state.faceLandmarker.detectForVideo(video, now);
    const handResult = state.handLandmarker.detectForVideo(video, now);
    const face = faceResult.faceLandmarks?.[0];
    const hand = handResult.landmarks?.[0];

    updateParallax(face);
    updateHandInteraction(hand);
    drawHandPreview(hand);
    setStatus(hand ? "Mano detectada" : face ? "Cabeza detectada" : "Buscando gesto", true);
  }

  renderScene();
  requestAnimationFrame(tick);
}

buttons.forEach((button) => {
  button.addEventListener("click", () => {
    const messages = {
      details: "Detalle activado: el gesto ha pulsado el boton.",
      next: "Siguiente activado: click recibido.",
    };

    showToast(messages[button.dataset.action] ?? "Boton activado.");

    button.animate(
      [
        { transform: "translateY(0) scale(1)" },
        { transform: "translateY(0) scale(0.94)" },
        { transform: "translateY(0) scale(1)" },
      ],
      { duration: 220, easing: "ease-out" },
    );
  });
});

async function init() {
  try {
    setStatus("Cargando fisicas");
    await setupPhysics();
    setStatus("Cargando escena");
    await setupThreeScene();
    renderScene();
    requestAnimationFrame(tick);
    setStatus("Pidiendo camara");
    await setupCamera();
    setStatus("Cargando tracking");
    await setupTracking();
    setStatus("Tracking listo", true);
  } catch (error) {
    console.error(error);
    setStatus(error?.message ?? "No se pudo iniciar");
  }
}

init();
