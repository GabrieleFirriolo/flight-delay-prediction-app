const fallbackEndpoint = "http://127.0.0.1:8000/agent/chat";

export const chatConfig = {
  endpoint: process.env.NEXT_PUBLIC_CHAT_ENDPOINT ?? fallbackEndpoint,
  samplePayload: {
    month: 12,
    day_of_week: 5,
    scheduled_departure_hour: 20,
    scheduled_arrival_hour: 23,
    origin_airport_code: "JFK",
    destination_airport_code: "MHT",
    airline_code: "AA",
    distance: 199.2
  },
} as const;

export type ChatConfig = typeof chatConfig;
