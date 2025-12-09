"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PaperAirplaneIcon } from "@heroicons/react/24/solid";
import { chatConfig } from "../config/chat";

type ChatMessage = {
  id: string;
  role: "user" | "agent";
  content: string;
  createdAt: string;
  isStreaming?: boolean;
};

type ApiResponse = {
  reply?: string;
  slots?: Record<string, unknown>;
  prediction?: Record<string, unknown> | null;
};

const SESSION_STORAGE_KEY = "flightdeck_session_id";

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return crypto.randomUUID();
  
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

function useTypewriter(text: string, baseSpeed: number = 20) {
  const [displayedText, setDisplayedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!text) {
      setDisplayedText("");
      setIsComplete(true);
      return;
    }

    setDisplayedText("");
    setIsComplete(false);
    let index = 0;
    let cancelled = false;

    const typeNextChar = () => {
      if (cancelled) return;
      if (index < text.length) {
        setDisplayedText(text.slice(0, index + 1));
        const char = text[index];
        index++;

        // Variable delay for natural typing feel
        let delay = baseSpeed;
        if (char === "." || char === "!" || char === "?") {
          delay = baseSpeed * 6;
        } else if (char === ",") {
          delay = baseSpeed * 2;
        } else if (char === " ") {
          delay = baseSpeed * 0.3;
        } else {
          delay = baseSpeed + Math.random() * baseSpeed * 0.3;
        }

        timeoutRef.current = setTimeout(typeNextChar, delay);
      } else {
        setIsComplete(true);
      }
    };

    timeoutRef.current = setTimeout(typeNextChar, baseSpeed);
    return () => {
      cancelled = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [text, baseSpeed]);

  return { displayedText, isComplete };
}

function StreamingMessage({ content, onComplete }: { content: string; onComplete: () => void }) {
  const { displayedText, isComplete } = useTypewriter(content, 15);

  useEffect(() => {
    if (isComplete) {
      onComplete();
    }
  }, [isComplete, onComplete]);

  return (
    <span>
      {displayedText}
      {!isComplete && <span className="animate-pulse">▊</span>}
    </span>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
      </div>
      <span className="text-sm text-slate-400">Agent is thinking...</span>
    </div>
  );
}

export default function Home() {
  const endpoint = chatConfig.endpoint;
  const [sessionId, setSessionId] = useState<string>("");
  const [messageDraft, setMessageDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>(() => [
    {
      id: "seed",
      role: "agent",
      content:
        "Agent console online. Send your flight details so we can predict potential delays.",
      createdAt: new Date().toISOString(),
    },
  ]);
  const [isSending, setIsSending] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<"idle" | "ok" | "error">("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // Initialize session ID on mount
  useEffect(() => {
    setSessionId(getOrCreateSessionId());
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [isLoading, messages]);

  const handleStreamComplete = useCallback((messageId: string) => {
    setStreamingMessageId(null);
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, isStreaming: false } : msg
      )
    );
  }, []);

  const sendMessage = async () => {
    if (!messageDraft.trim() || isSending) return;
    
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: messageDraft.trim(),
      createdAt: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setMessageDraft("");
    setIsSending(true);
    setIsLoading(true);
    setStatus("idle");
    setStatusMessage("");

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Agent endpoint responded with ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      const reply = data.reply ?? "No details returned from the prediction agent.";

      const agentMessageId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        {
          id: agentMessageId,
          role: "agent",
          content: reply,
          createdAt: new Date().toISOString(),
          isStreaming: true,
        },
      ]);
      setStreamingMessageId(agentMessageId);
      setStatus("ok");
      setStatusMessage("Agent responded successfully.");
    } catch (error) {
      console.error(error);
      setStatus("error");
      setStatusMessage(
        error instanceof Error ? error.message : "Unexpected failure.",
      );
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "agent",
          content:
            "We lost connection to the prediction agent. Re-check the endpoint and try again.",
          createdAt: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsSending(false);
      setIsLoading(false);
    }
  };

  const resetSession = () => {
    const newSessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);
    setSessionId(newSessionId);
    setMessages([
      {
        id: crypto.randomUUID(),
        role: "agent",
        content:
          "Session reset. Send your flight details so we can predict potential delays.",
        createdAt: new Date().toISOString(),
      },
    ]);
    setStatus("idle");
    setStatusMessage("");
  };

  const samplePayload = useMemo(
    () => JSON.stringify(chatConfig.samplePayload, null, 2),
    [],
  );

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-gradient-to-b from-sky-100 via-white to-indigo-100">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-20 top-10 h-64 w-64 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute right-0 top-1/2 h-72 w-72 -translate-y-1/2 rounded-full bg-indigo-200/30 blur-[120px]" />
        <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-indigo-300/20 to-transparent" />
      </div>

      <main className="relative z-10 flex h-full w-full items-stretch justify-center px-4 py-6 sm:px-6 sm:py-8">
        <section className="relative flex w-full max-w-4xl flex-col rounded-[32px] border border-white/20 bg-white/95 shadow-[0_45px_140px_-60px] shadow-slate-900/60 backdrop-blur overflow-hidden">
          <div className="absolute right-6 top-6">
            <div className="group relative">
              <span className="cursor-default rounded-full bg-slate-900 px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.3em] text-white shadow-lg">
                payload example
              </span>
              <div className="pointer-events-none absolute right-0 top-10 hidden w-64 rounded-2xl border border-slate-200 bg-slate-900/95 p-4 text-[11px] text-sky-100 shadow-2xl transition group-hover:block">
                <p className="mb-2 font-semibold tracking-[0.2em] text-slate-200">
                  Sample Body
                </p>
                <pre className="max-h-64 overflow-x-auto whitespace-pre-wrap">
{`POST ${endpoint}
Headers { "Content-Type": "application/json" }
Body ${samplePayload}`}
                </pre>
              </div>
            </div>
          </div>

          <header className="space-y-4 border-b border-slate-100 px-8 pb-6 pt-16 sm:pt-12">
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                Flight Delay Prediction
              </p>
              <h1 className="text-3xl font-semibold text-slate-900">
                Flight Deck Console
              </h1>
              <p className="text-sm text-slate-500">
                Chat with our expert flight deck assistant to predict your flight delays in USA.
              </p>
            </div>
            <div className="flex items-center justify-between rounded-2xl border border-slate-100 bg-slate-50/60 px-4 py-2 text-[11px] font-mono text-slate-400">
              <span>
                Session:{" "}
                <span className="font-semibold text-slate-700">
                  {sessionId ? `${sessionId.slice(0, 8)}…` : "initializing"}
                </span>
              </span>
              <button
                onClick={resetSession}
                className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-1.5 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600 shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-50 hover:text-slate-900"
              >
                <span className="text-xs">↻</span>
                New Session
              </button>
            </div>
          </header>

          <div className="flex min-h-0 flex-1 flex-col">
            <div
              ref={logRef}
              className="flex-1 space-y-4 overflow-y-auto px-8 py-6 scrollbar-thin scrollbar-thumb-slate-200"
            >
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`max-w-xl rounded-2xl border px-5 py-3 text-sm shadow-sm ${
                    message.role === "user"
                      ? "ml-auto border-sky-200 bg-sky-50 text-sky-900"
                      : "border-slate-200 bg-white text-slate-800"
                  }`}
                >
                  <header className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.25em] text-slate-400">
                    <span>
                      {message.role === "user" ? "User" : "Agent"}
                    </span>
                    <time className="font-mono">
                      {new Date(message.createdAt).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </time>
                  </header>
                  <p className="text-base leading-relaxed">
                    {message.isStreaming && message.id === streamingMessageId ? (
                      <StreamingMessage
                        content={message.content}
                        onComplete={() => handleStreamComplete(message.id)}
                      />
                    ) : (
                      message.content
                    )}
                  </p>
                </article>
              ))}
              {isLoading && (
                <article className="max-w-xl rounded-2xl border border-slate-200 bg-white px-5 py-3 text-sm shadow-sm">
                  <header className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.25em] text-slate-400">
                    <span>Agent</span>
                  </header>
                  <LoadingSkeleton />
                </article>
              )}
            </div>
            <div className="border-t border-slate-100 bg-slate-50/60 px-8 py-4">
              <div className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-inner focus-within:border-sky-400">
                <textarea
                  className="h-20 flex-1 resize-none bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400"
                  placeholder="e.g. Hi, i have a flight tomorrow evening at 8pm from New York to Manchester with American Airlines "
                  value={messageDraft}
                  onChange={(event) => setMessageDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void sendMessage();
                    }
                  }}
                />
                <button
                  disabled={isSending}
                  onClick={sendMessage}
                  className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 to-indigo-500 text-white shadow-lg shadow-sky-200 transition hover:scale-[1.03] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <PaperAirplaneIcon className="h-5 w-5 -translate-y-[1px]" />
                </button>
              </div>
              {status !== "idle" && (
                <p
                  className={`mt-3 text-xs font-medium ${
                    status === "ok" ? "text-emerald-600" : "text-rose-600"
                  }`}
                >
                  {statusMessage}
                </p>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
