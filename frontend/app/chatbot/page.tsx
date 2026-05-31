"use client"

import React, { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Send, Bot, User, Loader2 } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "sonner"

type Message = {
  id: string
  content: string
  sender: "user" | "bot"
  timestamp: Date
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm FarmBot, your agricultural assistant. How can I help you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!inputValue.trim()) {
      toast.warning("Empty message", {
        description: "Please type your question first",
        position: "top-center",
      })
      return
    }

    // Add user message immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: "user",
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    try{
      const response = await fetch(`${process.env.NEXT_PUBLIC_ML_SERVER_URL}/api/farm-chatbot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputValue }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      const botMessage: Message = {
        id: Date.now().toString(),
        content: data.response || "I couldn't process your request. Please try again.",
        sender: "bot",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      console.error("Chat error:", error)
      
      const errorMessage: Message = {

        id: Date.now().toString(),
        content: "Sorry, I'm having trouble responding right now. Please try again later.",
        sender: "bot",
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, errorMessage])
      
      toast.error("Connection Error", {
        description: "Failed to connect to the chatbot service",
        position: "top-center",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  return (
    <div className="container py-8">
      <div className="max-w-3xl mx-auto">
        <Card className="h-[calc(100vh-10rem)] flex flex-col">
          <CardHeader className="pb-3 border-b">
            <div className="flex items-center gap-3">
              <Avatar>
                <AvatarFallback className="bg-lime-100">
                  <Bot className="h-5 w-5 text-lime-700" />
                </AvatarFallback>
              </Avatar>
              <div>
                <CardTitle className="text-lg">FarmBot Assistant</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {isLoading ? "Typing..." : "Online"}
                </p>
              </div>
            </div>
          </CardHeader>

          <CardContent className="flex-grow overflow-hidden p-0">
            <ScrollArea className="h-full w-full">
              <div className="p-4 space-y-4">
                {messages.map((message) => (
                  <div 
                    key={message.id} 
                    className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div className={`flex gap-3 max-w-[85%] ${message.sender === "user" ? "flex-row-reverse" : ""}`}>
                      <Avatar className="h-8 w-8 mt-1">
                        <AvatarFallback className={message.sender === "user" ? "bg-gray-200" : "bg-lime-100"}>
                          {message.sender === "user" ? (
                            <User className="h-4 w-4 text-gray-700" />
                          ) : (
                            <Bot className="h-4 w-4 text-lime-700" />
                          )}
                        </AvatarFallback>
                      </Avatar>

                      <div className={`flex flex-col ${message.sender === "user" ? "items-end" : "items-start"}`}>
                        <div
                          className={`rounded-lg p-3 ${
                            message.sender === "user" 
                              ? "bg-lime-600 text-white rounded-tr-none" 
                              : "bg-gray-100 text-gray-800 rounded-tl-none"
                          }`}
                        >
                          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {formatTime(message.timestamp)}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="flex gap-3 max-w-[85%]">
                      <Avatar className="h-8 w-8 mt-1">
                        <AvatarFallback className="bg-lime-100">
                          <Bot className="h-4 w-4 text-lime-700" />
                        </AvatarFallback>
                      </Avatar>
                      <div className="rounded-lg p-3 bg-gray-100 text-gray-800 rounded-tl-none">
                        <div className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin text-lime-700" />
                          <p className="text-sm">Thinking...</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
          </CardContent>

          <CardFooter className="border-t pt-4">
            <form onSubmit={handleSendMessage} className="w-full flex gap-2">
              <Input
                placeholder="Ask about crops, weather, or farming..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                className="flex-grow"
                disabled={isLoading}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!inputValue.trim() || isLoading}
                className="bg-lime-600 hover:bg-lime-700"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </form>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}
