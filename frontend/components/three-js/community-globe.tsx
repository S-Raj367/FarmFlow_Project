"use client"

import { useRef, useEffect, useState, useMemo } from "react"
import * as THREE from "three"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import { Suspense } from "react"
// Add import for procedural textures
import { createEarthTexture, createCloudsTexture } from "./procedural-texture"

interface CommunityGlobeProps {
  className?: string
  locations?: { lat: number; lng: number; size?: number; color?: string }[]
}

export default function CommunityGlobe({ className = "", locations = [] }: CommunityGlobeProps) {
  const [hovered, setHovered] = useState(false)

  // Generate some random locations if none provided
  const defaultLocations = useMemo(() => {
    if (locations.length > 0) return locations

    return Array.from({ length: 50 }, () => ({
      lat: Math.random() * 180 - 90,
      lng: Math.random() * 360 - 180,
      size: Math.random() * 0.5 + 0.5,
      color: ["#84cc16", "#65a30d", "#4d7c0f"][Math.floor(Math.random() * 3)],
    }))
  }, [locations])

  return (
    <div
      className={`relative rounded-lg overflow-hidden ${className}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="h-full">
        <Canvas shadows dpr={[1, 2]}>
          <Suspense fallback={null}>
            <Scene locations={defaultLocations} isInteractive={hovered} />
          </Suspense>
        </Canvas>
      </div>
    </div>
  )
}

function Scene({
  locations,
  isInteractive,
}: {
  locations: { lat: number; lng: number; size?: number; color?: string }[]
  isInteractive: boolean
}) {
  return (
    <>
      <color attach="background" args={["#f8fafc"]} />

      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={1} />

      <OrbitControls
        autoRotate={true}
        autoRotateSpeed={0.5}
        enableZoom={isInteractive}
        enablePan={isInteractive}
        maxDistance={10}
        minDistance={2}
      />

      <Globe locations={locations} />
    </>
  )
}

// Update the Globe function to use procedural textures
function Globe({
  locations,
}: {
  locations: { lat: number; lng: number; size?: number; color?: string }[]
}) {
  const earthRef = useRef<THREE.Mesh>(null)
  const markersRef = useRef<THREE.Group>(null)
  const [earthTexture, setEarthTexture] = useState<THREE.Texture | null>(null)
  const [cloudsTexture, setCloudsTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {
    // Create the textures on client side
    const earthTex = createEarthTexture()
    const cloudsTex = createCloudsTexture()

    setEarthTexture(earthTex)
    setCloudsTexture(cloudsTex)

    return () => {
      // Clean up
      earthTex.dispose()
      cloudsTex.dispose()
    }
  }, [])

  useFrame((state) => {
    if (markersRef.current) {
      // Rotate markers in the opposite direction to keep them fixed relative to the globe
      markersRef.current.rotation.y = -state.clock.elapsedTime * 0.05
    }
  })

  // Convert lat/lng to 3D coordinates on a sphere
  const latLngToVector3 = (lat: number, lng: number, radius: number) => {
    const phi = (90 - lat) * (Math.PI / 180)
    const theta = (lng + 180) * (Math.PI / 180)

    const x = -radius * Math.sin(phi) * Math.cos(theta)
    const y = radius * Math.cos(phi)
    const z = radius * Math.sin(phi) * Math.sin(theta)

    return new THREE.Vector3(x, y, z)
  }

  return (
    <group>
      {/* Earth */}
      <mesh ref={earthRef} castShadow receiveShadow>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial
          color="#1e88e5"
          specular={new THREE.Color("grey")}
          shininess={5}
          map={earthTexture || undefined}
        />
      </mesh>

      {/* Continents - simplified representation */}
      {/*<mesh scale={[1.001, 1.001, 1.001]}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshPhongMaterial
          color="#4ade80"
          transparent={true}
          opacity={0.6}
          wireframe={true}
        />
      </mesh>*/}

      {/* Clouds */}
      <mesh scale={[1.01, 1.01, 1.01]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshPhongMaterial
          color="white"
          transparent={true}
          opacity={0.3}
          depthWrite={false}
          map={cloudsTexture || undefined}
        />
      </mesh>

      {/* Location markers */}
      <group ref={markersRef}>
        {locations.map((location, i) => {
          const position = latLngToVector3(location.lat, location.lng, 1.02)
          const size = location.size || 1
          const color = location.color || "#84cc16"

          return (
            <mesh key={i} position={position} castShadow>
              <sphereGeometry args={[0.02 * size, 16, 16]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
            </mesh>
          )
        })}
      </group>
    </group>
  )
}

