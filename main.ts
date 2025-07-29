import fs from "fs";
import path from "path";
import { promisify } from "util";
import readline from "readline";
import axios from "axios";
import { config } from "dotenv";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { BUSINESS_IDEAS } from "./data/ideas";

config(); // Load environment variables

// Promisify file system operations
const mkdir = promisify(fs.mkdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const exists = promisify(fs.exists);

// LLM Provider Configuration
type LLMProvider = 'openai' | 'anthropic' | 'gemini';

interface LLMConfig {
	provider: LLMProvider;
	model: string;
	apiKey: string;
}

// Initialize LLM clients
const openaiClient = process.env.OPENAI_API_KEY ? new OpenAI({
	apiKey: process.env.OPENAI_API_KEY
}) : null;

const anthropicClient = process.env.ANTHROPIC_API_KEY ? new Anthropic({
	apiKey: process.env.ANTHROPIC_API_KEY
}) : null;

const geminiClient = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(
	process.env.GEMINI_API_KEY
) : null;

// Available providers and their models
const AVAILABLE_PROVIDERS = {
	openai: {
		name: 'OpenAI',
		models: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
		requiredEnv: 'OPENAI_API_KEY',
		client: openaiClient
	},
	anthropic: {
		name: 'Anthropic Claude',
		models: ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
		requiredEnv: 'ANTHROPIC_API_KEY',
		client: anthropicClient
	},
	gemini: {
		name: 'Google Gemini',
		models: ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
		requiredEnv: 'GEMINI_API_KEY',
		client: geminiClient
	}
} as const;

// Business ideas data structure

// Interface for the ebook structure
interface Ebook {
	title: string;
	category: string;
	plan: string;
	chapters: {
		title: string;
		content: string;
		illustrationPrompt: string;
		illustrationPath?: string;
	}[];
	markdownContent?: string;
}

// CLI interface
const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});

// Helper functions
const question = (query: string): Promise<string> => {
	return new Promise((resolve) => rl.question(query, resolve));
};

const slugify = (text: string): string => {
	return text
		.toString()
		.toLowerCase()
		.replace(/\s+/g, "-") // Replace spaces with -
		.replace(/[^\w\-]+/g, "") // Remove all non-word chars
		.replace(/\-\-+/g, "-") // Replace multiple - with single -
		.replace(/^-+/, "") // Trim - from start of text
		.replace(/-+$/, ""); // Trim - from end of text
};

const generatePlanPrompt = (title: string): string => {
	return `
  Tu es un expert en entrepreneuriat à petit budget, spécialisé pour le marché ivoirien. 
  Je veux créer un ebook intitulé : "${title}"
  
  Génère un plan détaillé (8 à 12 chapitres) pour ce livre, avec des sous-sections pratiques.
  Pour chaque chapitre, suggère une idée d'illustration (ex : infographie, photo, schéma).
  
  Structure type :
  - Introduction
  - Chapitre 1 : Étude de marché
  - Chapitre 2 : Budget et financement
  - ... (autres chapitres pertinents)
  - Conclusion
  
  Retourne uniquement le plan au format Markdown, avec pour chaque chapitre :
  ### [Titre du chapitre]
  [Description du contenu]
  Illustration : [Description de l'illustration]
  `;
};

const generateIllustrationPrompt = (chapterTitle: string, ebookTitle: string): string => {
	return `
  Crée une illustration pour le chapitre "${chapterTitle}" d'un ebook sur "${ebookTitle}".
  
  Style souhaité :
  - Type : Infographie colorée / Photo réaliste
  - Éléments à inclure : Représentation visuelle du concept du chapitre
  - Couleurs vives (orange, vert, jaune)
  - Texte minimal (juste un titre si nécessaire)
  
  Décris précisément l'image à générer en 1-2 phrases.
  `;
};

const generateChapterPrompt = (chapterTitle: string, ebookTitle: string): string => {
	return `
  Rédige le chapitre "${chapterTitle}" pour un ebook intitulé "${ebookTitle}".
  
  Instructions :
  1. Longueur : 500 à 800 mots
  2. Structure :
     - Introduction (2-3 phrases)
     - Partie 1 : [Sous-thème 1] (étapes, exemples)
     - Partie 2 : [Sous-thème 2] (liste, conseils)
     - Encadré "Astuce" (1 conseil clé)
     - Encadré "Erreur à éviter" (1 piège courant)
     - Résumé (3 points clés)
  
  3. Ton :
     - Simple et direct
     - Utiliser "vous" pour impliquer le lecteur
     - Donner des exemples concrets avec estimations de prix
     - Adapter au marché ivoirien
  
  Retourne uniquement le contenu du chapitre, sans le titre.
  `;
};

// LLM API calls
const callLLM = async (prompt: string, config: LLMConfig): Promise<string> => {
	try {
		switch (config.provider) {
			case 'openai':
				if (!openaiClient) throw new Error('OpenAI client not initialized. Check OPENAI_API_KEY.');
				const openaiResponse = await openaiClient.chat.completions.create({
					model: config.model,
					messages: [{ role: "user", content: prompt }],
					temperature: 0.7,
				});
				return openaiResponse.choices[0].message.content?.trim() || '';

			case 'anthropic':
				if (!anthropicClient) throw new Error('Anthropic client not initialized. Check ANTHROPIC_API_KEY.');
				const anthropicResponse = await anthropicClient.messages.create({
					model: config.model,
					max_tokens: 4000,
					messages: [{ role: "user", content: prompt }],
					temperature: 0.7,
				});
				const content = anthropicResponse.content[0];
				return content.type === 'text' ? content.text.trim() : '';

			case 'gemini':
				if (!geminiClient) throw new Error('Gemini client not initialized. Check GEMINI_API_KEY.');
				const geminiModel = geminiClient.getGenerativeModel({ model: config.model });
				const geminiResponse = await geminiModel.generateContent(prompt);
				return geminiResponse.response.text().trim();

			default:
				throw new Error(`Unsupported provider: ${config.provider}`);
		}
	} catch (error) {
		console.error(`Error calling ${config.provider} API:`, error);
		throw error;
	}
};

const generateImage = async (prompt: string): Promise<string> => {
	try {
		// Currently only OpenAI DALL-E is supported for image generation
		if (!openaiClient) {
			throw new Error('OpenAI client not initialized. Image generation requires OPENAI_API_KEY.');
		}

		const response = await openaiClient.images.generate({
			prompt,
			n: 1,
			size: "512x512",
		});

		const imageUrl = response.data?.[0]?.url;
		if (!imageUrl) throw new Error('No image URL returned from OpenAI');

		// Download and save the image
		const imageResponse = await axios.get(imageUrl, { responseType: "arraybuffer" });
		return imageResponse.data;
	} catch (error) {
		console.error("Error generating image:", error);
		throw error;
	}
};

// Helper function to select LLM provider and model
const selectLLMProvider = async (): Promise<LLMConfig> => {
	// Check available providers
	const availableProviders = Object.entries(AVAILABLE_PROVIDERS).filter(
		([_, config]) => config.client !== null
	);

	if (availableProviders.length === 0) {
		throw new Error('No LLM providers available. Please set at least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)');
	}

	console.log('\n🤖 Available LLM Providers:');
	availableProviders.forEach(([key, config], i) => {
		console.log(`${i + 1}. ${config.name}`);
	});

	const providerChoice = await question('\nSelect LLM provider (number): ');
	const selectedProvider = availableProviders[parseInt(providerChoice) - 1];

	if (!selectedProvider) {
		throw new Error('Invalid provider selection');
	}

	const [providerKey, providerConfig] = selectedProvider;

	// Select model
	console.log(`\n📝 Available models for ${providerConfig.name}:`);
	providerConfig.models.forEach((model, i) => {
		console.log(`${i + 1}. ${model}`);
	});

	const modelChoice = await question('\nSelect model (number): ');
	const selectedModel = providerConfig.models[parseInt(modelChoice) - 1];

	if (!selectedModel) {
		throw new Error('Invalid model selection');
	}

	return {
		provider: providerKey as LLMProvider,
		model: selectedModel,
		apiKey: process.env[providerConfig.requiredEnv] || ''
	};
};

// Main ebook generation function
const generateEbook = async (title: string, category: string, llmConfig: LLMConfig) => {
	const slug = slugify(title);
	const categorySlug = slugify(category);
	const outputDir = path.join("output", categorySlug, slug);

	// Create output directory
	if (!(await exists(outputDir))) {
		await mkdir(outputDir, { recursive: true });
	}

	const ebook: Ebook = {
		title,
		category,
		plan: "",
		chapters: [],
	};

	console.log(`\n🚀 Starting generation for: ${title}`);

	// Step 1: Generate plan
	const planPath = path.join(outputDir, "1-title-chapters-plan.txt");
	if (await exists(planPath)) {
		console.log("↩️ Found existing plan, loading...");
		ebook.plan = await readFile(planPath, "utf-8");
	} else {
		console.log("📝 Generating plan...");
		ebook.plan = await callLLM(generatePlanPrompt(title), llmConfig);
		await writeFile(planPath, ebook.plan);
		console.log("✅ Plan generated and saved");
	}

	// Parse chapters from plan
	const chapterMatches = ebook.plan.matchAll(/### (.+?)\n([\s\S]+?)(?=###|$)/g);
	for (const match of chapterMatches) {
		const chapterTitle = match[1].trim();
		const chapterDescription = match[2].trim();

		// Extract illustration prompt from plan
		const illustrationMatch = chapterDescription.match(/Illustration : (.+)/);
		const illustrationPrompt = illustrationMatch ? illustrationMatch[1].trim() : await callLLM(generateIllustrationPrompt(chapterTitle, title), llmConfig);

		ebook.chapters.push({
			title: chapterTitle,
			content: "",
			illustrationPrompt,
		});
	}

	// Step 2: Generate illustrations and chapter contents
	for (let i = 0; i < ebook.chapters.length; i++) {
		const chapter = ebook.chapters[i];
		const chapterSlug = slugify(chapter.title);
		const chapterPrefix = `${i + 1}-${chapterSlug}`;

		console.log(`\n📖 Chapter ${i + 1}/${ebook.chapters.length}: ${chapter.title}`);

		// Generate illustration
		const illustrationPath = path.join(outputDir, `${chapterPrefix}.png`);
		if (await exists(illustrationPath)) {
			console.log("↩️ Found existing illustration, skipping...");
			chapter.illustrationPath = illustrationPath;
		} else {
			console.log("🎨 Generating illustration...");
			const imageData = await generateImage(chapter.illustrationPrompt);
			await writeFile(illustrationPath, imageData);
			chapter.illustrationPath = illustrationPath;
			console.log("✅ Illustration generated and saved");
		}

		// Generate chapter content
		const chapterPath = path.join(outputDir, `${chapterPrefix}.txt`);
		if (await exists(chapterPath)) {
			console.log("↩️ Found existing chapter content, loading...");
			chapter.content = await readFile(chapterPath, "utf-8");
		} else {
			console.log("✍️ Generating chapter content...");
			chapter.content = await callLLM(generateChapterPrompt(chapter.title, title), llmConfig);
			await writeFile(chapterPath, chapter.content);
			console.log("✅ Chapter content generated and saved");
		}

		// Pause point for verification
		if (i < ebook.chapters.length - 1) {
			const answer = await question("Continue to next chapter? (y/n) ");
			if (answer.toLowerCase() !== "y") {
				console.log("⏸️ Generation paused. You can resume later.");
				return;
			}
		}
	}

	// Step 3: Compile markdown
	console.log("\n📚 Compiling markdown...");
	let markdown = `# ${title}\n\n`;

	// Add introduction
	markdown += `## Introduction\n\n`;
	markdown += `Ce guide vous montrera comment ${title.toLowerCase()}.\n\n`;

	// Add chapters
	for (let i = 0; i < ebook.chapters.length; i++) {
		const chapter = ebook.chapters[i];
		const chapterSlug = slugify(chapter.title);
		const imagePath = `${i + 1}-${chapterSlug}.png`;

		markdown += `## Chapitre ${i + 1}: ${chapter.title}\n\n`;
		markdown += `![Illustration pour ${chapter.title}](${imagePath})\n\n`;
		markdown += `${chapter.content}\n\n`;
	}

	// Add conclusion
	markdown += `## Conclusion\n\n`;
	markdown += `Vous avez maintenant toutes les clés pour ${title.toLowerCase()}.\n\n`;

	// Save markdown
	const markdownPath = path.join(outputDir, `${slug}.md`);
	await writeFile(markdownPath, markdown);
	console.log(`✅ Markdown compiled and saved to ${markdownPath}`);

	return ebook;
};

// Main CLI function
const main = async () => {
	console.log("📘 Business Idea Ebook Generator CLI\n");

	// List categories
	console.log("📂 Categories:");
	const categories = Object.keys(BUSINESS_IDEAS);
	categories.forEach((cat, i) => {
		console.log(`${i + 1}. ${cat}`);
	});

	// Select category
	const catChoice = await question("\nSelect a category (number): ");
	const selectedCategory = categories[parseInt(catChoice) - 1];

	if (!selectedCategory) {
		console.error("Invalid category selection");
		process.exit(1);
	}

	// List ideas in category
	console.log(`\n💡 Ideas in ${selectedCategory}:`);
	const ideas = BUSINESS_IDEAS[selectedCategory];
	ideas.forEach((idea, i) => {
		console.log(`${i + 1}. ${idea}`);
	});

	// Select idea
	const ideaChoice = await question("\nSelect an idea (number) or enter custom title: ");
	let selectedTitle = "";

	if (isNaN(parseInt(ideaChoice))) {
		selectedTitle = ideaChoice;
	} else {
		selectedTitle = ideas[parseInt(ideaChoice) - 1];
	}

	if (!selectedTitle) {
		console.error("Invalid idea selection");
		process.exit(1);
	}

	// Select LLM provider and model
	let llmConfig: LLMConfig;
	try {
		llmConfig = await selectLLMProvider();
		console.log(`\n✅ Selected: ${AVAILABLE_PROVIDERS[llmConfig.provider].name} - ${llmConfig.model}`);
	} catch (error) {
		console.error("\n❌ Error selecting LLM provider:", error);
		rl.close();
		return;
	}

	// Start generation
	try {
		await generateEbook(selectedTitle, selectedCategory, llmConfig);
		console.log("\n🎉 Ebook generation completed successfully!");
	} catch (error) {
		console.error("\n❌ Error during ebook generation:", error);
	} finally {
		rl.close();
	}
};

// Run the program
main().catch(console.error);
