import * as THREE from "three"

// Create a procedural grass texture
export function createGrassTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 512
  canvas.height = 512
  const context = canvas.getContext("2d")

  if (!context) return new THREE.Texture()

  // Fill background
  context.fillStyle = "#4ade80"
  context.fillRect(0, 0, canvas.width, canvas.height)

  // Add some variation
  for (let i = 0; i < 1000; i++) {
    const x = Math.random() * canvas.width
    const y = Math.random() * canvas.height
    const size = Math.random() * 3 + 1

    context.fillStyle = Math.random() > 0.5 ? "#65a30d" : "#84cc16"
    context.beginPath()
    context.arc(x, y, size, 0, Math.PI * 2)
    context.fill()
  }

  // Add some grass blades
  for (let i = 0; i < 200; i++) {
    const x = Math.random() * canvas.width
    const y = Math.random() * canvas.height
    const height = Math.random() * 20 + 5
    const width = Math.random() * 2 + 1

    context.fillStyle = Math.random() > 0.5 ? "#84cc16" : "#65a30d"
    context.fillRect(x, y, width, height)
  }

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping
  texture.repeat.set(10, 10)

  return texture
}

// Create a procedural earth texture
export function createEarthTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 512
  const context = canvas.getContext("2d")

  if (!context) return new THREE.Texture()

  // Fill with ocean blue
  context.fillStyle = "#1e88e5"
  context.fillRect(0, 0, canvas.width, canvas.height)

  // Add some continents (very simplified)
  context.fillStyle = "#4ade80"

  // North America
  context.beginPath()
  context.moveTo(200, 150)
  context.lineTo(300, 120)
  context.lineTo(350, 200)
  context.lineTo(300, 250)
  context.lineTo(250, 230)
  context.fill()

  // South America
  context.beginPath()
  context.moveTo(300, 280)
  context.lineTo(350, 300)
  context.lineTo(330, 400)
  context.lineTo(280, 380)
  context.fill()

  // Europe and Africa
  context.beginPath()
  context.moveTo(500, 150)
  context.lineTo(550, 140)
  context.lineTo(580, 200)
  context.lineTo(550, 350)
  context.lineTo(500, 300)
  context.fill()

  // Asia and Australia
  context.beginPath()
  context.moveTo(600, 150)
  context.lineTo(750, 180)
  context.lineTo(700, 250)
  context.lineTo(750, 350)
  context.lineTo(650, 300)
  context.fill()

  const texture = new THREE.CanvasTexture(canvas)

  return texture
}

// Create a procedural clouds texture
export function createCloudsTexture() {
  const canvas = document.createElement("canvas")
  canvas.width = 1024
  canvas.height = 512
  const context = canvas.getContext("2d")

  if (!context) return new THREE.Texture()

  // Clear canvas
  context.fillStyle = "black"
  context.fillRect(0, 0, canvas.width, canvas.height)

  // Add some clouds
  context.fillStyle = "white"
  for (let i = 0; i < 100; i++) {
    const x = Math.random() * canvas.width
    const y = Math.random() * canvas.height
    const radius = Math.random() * 30 + 10

    context.beginPath()
    context.arc(x, y, radius, 0, Math.PI * 2)
    context.fill()

    // Add some variation to the clouds
    for (let j = 0; j < 5; j++) {
      const cloudX = x + (Math.random() * 40 - 20)
      const cloudY = y + (Math.random() * 40 - 20)
      const cloudRadius = Math.random() * 20 + 5

      context.beginPath()
      context.arc(cloudX, cloudY, cloudRadius, 0, Math.PI * 2)
      context.fill()
    }
  }

  const texture = new THREE.CanvasTexture(canvas)

  return texture
}

