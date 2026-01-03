"use client"

import { useRef, useEffect, useState } from "react"
import * as THREE from "three"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Environment, Float, PerspectiveCamera } from '@react-three/drei'
import { Suspense } from "react"
import { useMobile } from "@/hooks/use-mobile"
// Add import for procedural textures
import { createGrassTexture } from "./procedural-texture"

export default function HeroBackground() {
  const isMobile = useMobile()
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    // Set loaded after a small delay to allow for smooth fade-in
    const timer = setTimeout(() => setLoaded(true), 100)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className={`absolute inset-0 transition-opacity duration-1000 ${loaded ? "opacity-100" : "opacity-0"}`}>
      <Canvas shadows dpr={[1, 2]}>
        <Suspense fallback={null}>
          <Scene isMobile={isMobile} />
          <Environment preset="sunset" />
        </Suspense>
      </Canvas>
    </div>
  )
}

function Scene({ isMobile }: { isMobile: boolean }) {
  const { camera } = useThree()

  useEffect(() => {
    // Adjust camera position based on screen size
    if (isMobile) {
      camera.position.set(0, 2, 10)
    } else {
      camera.position.set(0, 2, 8)
    }
  }, [camera, isMobile])

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 2, 8]} fov={40} />
      <color attach="background" args={["#e9f5e1"]} />
      <fog attach="fog" args={["#e9f5e1", 5, 20]} />

      <ambientLight intensity={0.5} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={1}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-far={20}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
      />

      <group position={[0, -1, 0]}>
        <Terrain />
        <FloatingTractor position={[-2, 0.5, 1]} scale={0.5} rotation={[0, Math.PI / 4, 0]} />
        <FloatingPlant position={[2, 0.3, 0]} scale={0.4} />
        <FloatingCorn position={[0, 0.3, -2]} scale={0.3} />
        <Clouds />
      </group>
    </>
  )
}

// Update the Terrain function to use procedural texture
function Terrain() {
  const [grassTexture, setGrassTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    // Create the grass texture on client side
    const texture = createGrassTexture()
    setGrassTexture(texture)

    return () => {
      // Clean up
      texture.dispose()
    }
  }, [])

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[100, 100, 64, 64]} />
      <meshStandardMaterial color="#4ade80" roughness={0.8} metalness={0.1} map={grassTexture || undefined} />
    </mesh>
  )
}

function FloatingTractor({ position, scale, rotation }: any) {
  const group = useRef<THREE.Group>(null)

  // Simulate loading a GLTF model
  useFrame((state) => {
    if (group.current) {
      group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1 + (rotation?.[1] || 0)
      group.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1 + position[1]
    }
  })

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
      <group ref={group} position={position} scale={scale} rotation={rotation}>
        {/* Simplified tractor shape */}
        <mesh castShadow>
          <boxGeometry args={[2, 1, 3]} />
          <meshStandardMaterial color="#e11d48" />
        </mesh>
        <mesh position={[0, 0.8, 0]} castShadow>
          <boxGeometry args={[1.5, 0.6, 1.5]} />
          <meshStandardMaterial color="#e11d48" />
        </mesh>
        {/* Wheels */}
        <mesh position={[-1, -0.5, 1]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.5, 0.5, 0.3, 16]} />
          <meshStandardMaterial color="#1e293b" />
        </mesh>
        <mesh position={[1, -0.5, 1]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.5, 0.5, 0.3, 16]} />
          <meshStandardMaterial color="#1e293b" />
        </mesh>
        <mesh position={[-1, -0.5, -1]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.5, 0.5, 0.3, 16]} />
          <meshStandardMaterial color="#1e293b" />
        </mesh>
        <mesh position={[1, -0.5, -1]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.5, 0.5, 0.3, 16]} />
          <meshStandardMaterial color="#1e293b" />
        </mesh>
      </group>
    </Float>
  )
}

function FloatingPlant({ position, scale }: any) {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (group.current) {
      group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1
      group.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1 + position[1]
    }
  })

  return (
    <Float speed={1.5} rotationIntensity={0.2} floatIntensity={0.5}>
      <group ref={group} position={position} scale={scale}>
        {/* Stem */}
        <mesh position={[0, 1, 0]} castShadow>
          <cylinderGeometry args={[0.1, 0.2, 2, 8]} />
          <meshStandardMaterial color="#65a30d" />
        </mesh>
        {/* Leaves */}
        <mesh position={[0, 1.5, 0]} rotation={[0, 0, Math.PI / 4]} castShadow>
          <coneGeometry args={[0.8, 1.5, 4]} />
          <meshStandardMaterial color="#84cc16" side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, 2, 0]} rotation={[0, Math.PI / 4, Math.PI / 4]} castShadow>
          <coneGeometry args={[0.6, 1.2, 4]} />
          <meshStandardMaterial color="#84cc16" side={THREE.DoubleSide} />
        </mesh>
        {/* Pot */}
        <mesh position={[0, 0, 0]} castShadow>
          <cylinderGeometry args={[0.5, 0.3, 1, 16]} />
          <meshStandardMaterial color="#a16207" />
        </mesh>
      </group>
    </Float>
  )
}

function FloatingCorn({ position, scale }: any) {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (group.current) {
      group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.1
      group.current.position.y = Math.sin(state.clock.elapsedTime * 0.4) * 0.1 + position[1]
    }
  })

  return (
    <Float speed={1} rotationIntensity={0.1} floatIntensity={0.3}>
      <group ref={group} position={position} scale={scale}>
        {/* Stem */}
        <mesh position={[0, 2, 0]} castShadow>
          <cylinderGeometry args={[0.1, 0.2, 4, 8]} />
          <meshStandardMaterial color="#65a30d" />
        </mesh>
        {/* Corn */}
        <mesh position={[0.3, 2, 0]} rotation={[0, 0, -Math.PI / 6]} castShadow>
          <capsuleGeometry args={[0.3, 1, 8, 16]} />
          <meshStandardMaterial color="#facc15" />
        </mesh>
        {/* Leaves */}
        <mesh position={[0, 2, 0]} rotation={[0, 0, Math.PI / 3]} castShadow>
          <boxGeometry args={[0.1, 2, 0.5]} />
          <meshStandardMaterial color="#84cc16" />
        </mesh>
        <mesh position={[0, 3, 0]} rotation={[0, 0, -Math.PI / 3]} castShadow>
          <boxGeometry args={[0.1, 2, 0.5]} />
          <meshStandardMaterial color="#84cc16" />
        </mesh>
        <mesh position={[0, 2.5, 0]} rotation={[Math.PI / 3, 0, 0]} castShadow>
          <boxGeometry args={[0.1, 2, 0.5]} />
          <meshStandardMaterial color="#84cc16" />
        </mesh>
      </group>
    </Float>
  )
}

function Clouds() {
  const group = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (group.current) {
      group.current.position.x = Math.sin(state.clock.elapsedTime * 0.05) * 3
    }
  })

  return (
    <group ref={group} position={[0, 5, -5]}>
      <Cloud position={[-4, 0, 0]} scale={1} />
      <Cloud position={[0, 1, -2]} scale={1.5} />
      <Cloud position={[4, 0.5, -1]} scale={1.2} />
    </group>
  )
}

function Cloud({ position, scale }: any) {
  return (
    <group position={position} scale={scale}>
      <mesh castShadow>
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>
      <mesh position={[1, -0.2, 0]} castShadow>
        <sphereGeometry args={[0.8, 16, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>
      <mesh position={[-1, -0.1, 0]} castShadow>
        <sphereGeometry args={[0.7, 16, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>
      <mesh position={[0, 0.5, 0]} castShadow>
        <sphereGeometry args={[0.7, 16, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>
    </group>
  )
}

