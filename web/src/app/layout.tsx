import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { HealthProvider } from "@/components/HealthProvider";
import { ModelGate } from "@/components/ModelGate";
import { ModelTag } from "@/components/ModelTag";
import { TopBar } from "@/components/TopBar";
import "./globals.css";
import { t } from "@/lib/copy";
import s from "./layout.module.css";

const sans = Geist({ subsets: ["latin"], variable: "--font-sans" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: { default: "NeuronLens", template: "%s · NeuronLens" },
  description: "See which concepts a language model's learned features were active on, and test them by steering.",
};

export const viewport: Viewport = { themeColor: "#000000", colorScheme: "dark" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <HealthProvider>
          <TopBar />
          <main id="main">
            <ModelGate>{children}</ModelGate>
          </main>
          <footer className={s.footer}>
            <div className={s.footInner}>
              <span>{t("note.limit")}</span>
              <ModelTag />
            </div>
          </footer>
        </HealthProvider>
      </body>
    </html>
  );
}
