#include <sstream>
#include <string>
#include <vector> //for stack + info storage
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdio.h> //for getting file fd
#include "clang/Lex/Lexer.h" //for getting source code from the AST.
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/AST.h"
#include "clang/AST/ParentMap.h"//for getting parent of a node
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_MODE false
#define HALF2_OVERLOAD_HEADER "half2_operator_overload.cuh"
#define HALF_OVERLOAD_HEADER "half_operator_overload.cuh"
//define half2 api here (with open parenthesis), in case Nvidia may change the API later.
#define HALF2_ADD "__hadd2("
#define HALF2_SUB "__hsub2("
#define HALF2_MUL "__hmul2("
#define HALF2_DIV "h2div("
#define FLOAT2HALF2 "__float2half2_rn("

#define HALF2_NEG "h2neg(" //math api doesn't have, need to append the implementation to the code.
//compare ops
#define HALF2_LT "__hlt2("
#define HALF2_LE "__hle2("
#define HALF2_EQ "__heq2("
#define HALF2_NE "__hne2("
#define HALF2_GT "__hgt2("
#define HALF2_GE "__hge2("
//math funcs
#define HALF2_COS "h2cos("
#define HALF2_SIN "h2sin("
#define HALF2_EXP "h2exp("
#define HALF2_EXP10 "h2exp10("
#define HALF2_EXP2 "h2exp2("
#define HALF2_LOG "h2log("
#define HALF2_LOG10 "h2log10("
#define HALF2_LOG2 "h2log2("
#define HALF2_RSQRT "h2rsqrt("
#define HALF2_SQRT "h2sqrt("
#define HALF2_ABS "h2abs("

//half version without simd 
#define HALF_ADD "__hadd("
#define HALF_SUB "__hsub("
#define HALF_MUL "__hmul("
#define HALF_DIV "hdiv("
#define FLOAT2HALF "__float2half("

#define HALF_NEG "hneg(" //math api doesn't have, need to append the implementation to the code.
//compare ops
#define HALF_LT "__hlt("
#define HALF_LE "__hle("
#define HALF_EQ "__heq("
#define HALF_NE "__hne("
#define HALF_GT "__hgt("
#define HALF_GE "__hge("
//math funcs
#define HALF_COS "hcos("
#define HALF_SIN "hsin("
#define HALF_EXP "hexp("
#define HALF_EXP10 "hexp10("
#define HALF_EXP2 "hexp2("
#define HALF_LOG "hlog("
#define HALF_LOG10 "hlog10("
#define HALF_LOG2 "hlog2("
#define HALF_RSQRT "hrsqrt("
#define HALF_SQRT "hsqrt("
#define HALF_ABS "habs("
//end half version


//mathfunc float
#define FLOAT_DIV "fdividef("
#define FLOAT_COS "cosf("
#define FLOAT_SIN "sinf("
#define FLOAT_EXP "expf("
#define FLOAT_EXP10 "exp10f("
#define FLOAT_EXP2 "exp2f("
#define FLOAT_LOG "logf("
#define FLOAT_LOG10 "log10f("
#define FLOAT_LOG2 "log2f("
#define FLOAT_RSQRT "rsqrtf("
#define FLOAT_SQRT "sqrtf("
#define FLOAT_ABS "fabsf("
//mathfunc double
#define DOUBLE_COS "cos("
#define DOUBLE_SIN "sin("
#define DOUBLE_EXP "exp("
#define DOUBLE_EXP10 "exp10("
#define DOUBLE_EXP2 "exp2("
#define DOUBLE_LOG "log("
#define DOUBLE_LOG10 "log10("
#define DOUBLE_LOG2 "log2("
#define DOUBLE_RSQRT "rsqrt("
#define DOUBLE_SQRT "sqrt("
#define DOUBLE_ABS "fabs("

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;
using namespace std;
static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");


//global variables to store variables info (type )/ can be read from a config file or be automatically detected by the tool
 //rule to store vars: pair them with their parent function varname*****function_name. global vars => pair *****global
  vector<pair<string,string> > IndexVars; //variables used for thread index
  vector<pair<string,string> > NoThreadVars; //possible var used for identifying number of threads in device code: if (idx< n) {do sthing} else {do nothing}
  vector<pair<string,string> > half2Vars; // variable are in half2 precision. due to complication of conversion from floating point literal to half2. there may be some constant/var decl need to be in float.

  vector<pair<int,int> > if_stmt_linenumber; // <start_line,end_line> of an if/else condition, to keep track of divergence cuda code
  vector<pair<int,IfStmt*> > if_stmt_endline;// <end_line, if_stmt pointer> of an if/else condition, to write post_process masking code & update translator's context

  bool func_overload_mode = true; //in func_overload_mode, we use func overload header to rewrite half2 & half operations
  bool  half2_mode = true;
  string half_type_string = "__half2";
  bool first_statement = true;
// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  MyASTVisitor(Rewriter &R,ASTContext *Context) : TheRewriter(R), Context(Context), CurrentFunc(nullptr), CudaCode(false){
	  currentIfLineNumber = make_pair(0,0);
	  currentElseLineNumber = make_pair(0,0);
	  insideIfCond = false;
	  insideElseBranch = false;

	  }

	
	void removeCharsFromString( string &str, char* charsToRemove ) {
	   for ( unsigned int i = 0; i < strlen(charsToRemove); ++i ) {
		  str.erase( remove(str.begin(), str.end(), charsToRemove[i]), str.end() );
	   }
	}
	bool IsFloatingPointType(string typeStr){
		return (typeStr.find("double") != string::npos) || (typeStr.find("float") != string::npos);
	}
	bool IsIntegerType(string typeStr){
		return (typeStr.find("int") != string::npos) || (typeStr.find("long") != string::npos);
	}
	bool IsThreadIDRelated(string expr){
		return (expr.find("blockIdx") != string::npos) || (expr.find("blockDim") != string::npos)|| (expr.find("threadIdx") != string::npos);
	}
	bool IsHalf2Var(string varName){
		pair<string,string> elementToFind1;
		pair<string,string> elementToFind2;
		pair<string,string> elementToFind3;
		elementToFind1 = make_pair(varName, "");
		elementToFind3 = make_pair(varName, varName); //for function
		if (CurrentFunc!=nullptr)
			elementToFind2 = make_pair(varName, CurrentFunc->getNameAsString());
		else elementToFind2 = elementToFind1;

		if((std::find(half2Vars.begin(), half2Vars.end(), elementToFind1) != half2Vars.end())  ||(std::find(half2Vars.begin(), half2Vars.end(), elementToFind2) != half2Vars.end()) ||(std::find(half2Vars.begin(), half2Vars.end(), elementToFind3) != half2Vars.end()))
			return true ;

		else return false;

	}
	
		bool IsThreadIdVar(string varName){
		pair<string,string> elementToFind1;
		pair<string,string> elementToFind2;
		pair<string,string> elementToFind3;
		elementToFind1 = make_pair(varName, "");
		if (CurrentFunc!=nullptr)
			elementToFind2 = make_pair(varName, CurrentFunc->getNameAsString());
		else elementToFind2 = elementToFind1;

		if((std::find(IndexVars.begin(), IndexVars.end(), elementToFind1) != IndexVars.end())  ||(std::find(IndexVars.begin(), IndexVars.end(), elementToFind2) != IndexVars.end()))
			return true ;

		else return false;

	}
	bool isProcessingFunction(string functionName){ //there are many aux function in cuda headers & clang headers for cuda. must filter it out, only process our target functions 
		
		return IsHalf2Var(functionName);
	}
	bool IsHalf2Expr(Expr* expr){
				//cout<<"IsHalf2Expr " <<"\n";
				
				Expr* exprIgnoreParensCasts = expr->IgnoreImpCasts()->IgnoreParens ();
				//exprIgnoreParensCasts->dump();
				if(isa <ArraySubscriptExpr> (exprIgnoreParensCasts ) ||  isa <DeclRefExpr>(exprIgnoreParensCasts ) || isa <UnaryOperator>(exprIgnoreParensCasts ) ){
				//	cout<<"inside if " <<"\n";
					DeclRefExpr* declExpr = nullptr;
					if (isa <ArraySubscriptExpr> (exprIgnoreParensCasts)){
						ArraySubscriptExpr* subscriptExpr = cast<ArraySubscriptExpr>(exprIgnoreParensCasts);
						if (isa <ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts()))
							subscriptExpr = cast<ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts());
						if (isa <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts()))
							declExpr = cast <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts());
					} else if (isa <UnaryOperator> (exprIgnoreParensCasts)){ //only the case *var  when var is a pointer to a floating var

							UnaryOperator * unaryOp =  cast <UnaryOperator>(exprIgnoreParensCasts);
							if(unaryOp->getOpcode() == UO_Deref)
								exprIgnoreParensCasts = unaryOp->getSubExpr()->IgnoreCasts()->IgnoreImpCasts ();
						}
					if(isa <DeclRefExpr>(exprIgnoreParensCasts))
						declExpr = cast <DeclRefExpr> (exprIgnoreParensCasts);
					//subscriptExpr->dump();
					if (declExpr!=nullptr){
						string baseSrc = declExpr-> getDecl ()-> getNameAsString();
					//	cout<<"IsHalf2Expr not " << baseSrc<< " Is half2: "<<IsHalf2Var(baseSrc) <<"\n";

							//if (IsHalf2Var(baseSrc)&& CurrentFunc==nullptr) return true;
							//if (!IsHalf2Var(baseSrc)) return true; //ofc
							return IsHalf2Var(baseSrc);
					}
				}
				return false;
		}
	string getSourceTextFromSourceRange(SourceRange* sourceRange ){
		return Lexer::getSourceText(CharSourceRange::getCharRange(*sourceRange),Context->getSourceManager(),LangOptions(), 0);
		}
	SourceRange* getSourceRangeIgnoreArrayRef(Expr* expr){ //return a source range of array name, in this way, we can retrieve it later e.g. expr = z[i] -> return sourcerange of the character "z"
				if(DEBUG_MODE)
				cout<<"getSourceRangeIgnoreArrayRef " <<"\n";
				
				Expr* exprIgnoreParensCasts = expr->IgnoreImpCasts()->IgnoreParens ();
				//exprIgnoreParensCasts->dump();
				if(isa <ArraySubscriptExpr> (exprIgnoreParensCasts ) ||  isa <DeclRefExpr>(exprIgnoreParensCasts ) || isa <UnaryOperator>(exprIgnoreParensCasts ) ){
					//cout<<"inside if " <<"\n";
					DeclRefExpr* declExpr = nullptr;
					if (isa <ArraySubscriptExpr> (exprIgnoreParensCasts)){
						ArraySubscriptExpr* subscriptExpr = cast<ArraySubscriptExpr>(exprIgnoreParensCasts);
						if (isa <ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts()))
							subscriptExpr = cast<ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts());
						if (isa <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts()))
							declExpr = cast <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts());
					} else if (isa <UnaryOperator> (exprIgnoreParensCasts)){ //only the case *var  when var is a pointer to a floating var

							UnaryOperator * unaryOp =  cast <UnaryOperator>(exprIgnoreParensCasts);
							if(unaryOp->getOpcode() == UO_Deref)
								exprIgnoreParensCasts = unaryOp->getSubExpr()->IgnoreCasts()->IgnoreImpCasts ();
						}
					if(isa <DeclRefExpr>(exprIgnoreParensCasts))
						declExpr = cast <DeclRefExpr> (exprIgnoreParensCasts);
					//subscriptExpr->dump();
					if (declExpr!=nullptr){
						//string baseSrc = declExpr-> getDecl ()-> getNameAsString();
						
						SourceLocation startLoc = declExpr->getLocStart ();
						SourceLocation endLoc = declExpr->getLocEnd ();
						
					//	cout<<"IsHalf2Expr not " << baseSrc<< " Is half2: "<<IsHalf2Var(baseSrc) <<"\n";

							//if (IsHalf2Var(baseSrc)&& CurrentFunc==nullptr) return true;
							//if (!IsHalf2Var(baseSrc)) return true; //ofc
							return new SourceRange (startLoc,endLoc );
						
					} else return nullptr;
				} else{ //no special case, treat like usual  z_val => return range of z_val;
						return new SourceRange(exprIgnoreParensCasts->getLocStart(), exprIgnoreParensCasts->getLocEnd());
				}
				
		}	
	void RewriteFunctionCall(CallExpr* E){
		if(func_overload_mode)
			return;
		string funcName = Lexer::getSourceText(CharSourceRange::getCharRange(E->getLocStart(), E->getArg(0)->getLocStart()),Context->getSourceManager(),LangOptions(), 0);
		int replaceLength = funcName.length();
		removeCharsFromString( funcName, "_ " );
		if(DEBUG_MODE)
			cout<<"rewriteFunccall "<<funcName<<" " <<replaceLength<<"\n";
		string replaceString = "";
		if (funcName == FLOAT_DIV)
			replaceString = half2_mode?HALF2_DIV:HALF_DIV;
		else if (funcName==FLOAT_COS || funcName==DOUBLE_COS)
			replaceString = half2_mode?HALF2_COS:HALF_COS;
		else if (funcName==FLOAT_SIN || funcName==DOUBLE_SIN)
			replaceString = half2_mode?HALF2_SIN:HALF_SIN;
		else if (funcName==FLOAT_EXP || funcName==DOUBLE_EXP)
			replaceString = half2_mode?HALF2_EXP:HALF_EXP;
		else if (funcName==FLOAT_EXP10 || funcName==DOUBLE_EXP10)
			replaceString = half2_mode?HALF2_EXP10:HALF_EXP10;
		else if (funcName==FLOAT_EXP2 || funcName==DOUBLE_EXP2)
			replaceString = half2_mode?HALF2_EXP2:HALF_EXP2;
		else if (funcName==FLOAT_LOG || funcName==DOUBLE_LOG)
			replaceString = half2_mode?HALF2_LOG:HALF_LOG;
		else if (funcName==FLOAT_LOG10 || funcName==DOUBLE_LOG10)
			replaceString = half2_mode?HALF2_LOG10:HALF_LOG10;
		else if (funcName==FLOAT_LOG2 || funcName==DOUBLE_LOG2)
			replaceString = half2_mode?HALF2_LOG2:HALF_LOG2;
		else if (funcName==FLOAT_RSQRT  || funcName==DOUBLE_RSQRT)
			replaceString = half2_mode?HALF2_RSQRT:HALF_RSQRT;
		else if (funcName==FLOAT_SQRT || funcName==DOUBLE_SQRT)
			replaceString = half2_mode?HALF2_SQRT:HALF_SQRT;
		else if (funcName==FLOAT_ABS || funcName==DOUBLE_ABS)
			replaceString = half2_mode?HALF2_ABS:HALF_ABS;

		if (replaceString!=""){
			TheRewriter.ReplaceText(E->getLocStart(), replaceLength, replaceString);

			}

		return;
		}
	string insertStringBeforeChar(string target, string insert_str, string search_str){ //insert string before the first occurence of char
		string result_str = ""+ target;
		int insertPosition = result_str.find_first_of(search_str);
		if(insertPosition== -1)
			insertPosition = result_str.length();
		result_str.insert(insertPosition,insert_str );
		return result_str;
	}
	template <typename T>
	void insertToVectorIfNotExist(vector<T> &vector_x, T val){
		if (std::find(vector_x.begin(), vector_x.end(), val) == vector_x.end()) {
		  // someName not in name, add it
		  vector_x.push_back(val);
		}
		return;
	}
	string processAssignPreIff(IfStmt* ifStatement){
	//	cout <<"xxxxxxxxx"<<"processAssignPreIff"<<"\n";
		stringstream ssResult;
		for(string var : lhsMaskedInIffCond) {
			string temp_var = proccessArrayRefStr(var);
			ssResult << insertStringBeforeChar(temp_var,"_masked_"+ to_string(currentIfLineNumber.first), "[");
			ssResult <<  " = "<<var << ";\n";
		}
		if (!lhsMaskedInElseCond.empty()){
				for(string var : lhsMaskedInElseCond) {
				string temp_var = proccessArrayRefStr(var);
				ssResult << insertStringBeforeChar(temp_var,"_masked_"+ to_string(currentIfLineNumber.first), "[");
				ssResult <<  "_else = "<<var << ";\n";
				}			
			}
		if(DEBUG_MODE)
			cout<<ssResult.str() ; 
		//~ return "";
		return ssResult.str();
		}
	string processAssignPostIff(IfStmt* ifStatement){
		stringstream ssResult;
		for (int i = 0; i<2; i++){
			ssResult << "if((short*)&mask_if_"+ to_string(currentIfLineNumber.first) +"["<<i<<"]) { \n";
			for(string var : lhsMaskedInIffCond) {
				string temp_var = proccessArrayRefStr(var);
				ssResult << insertStringBeforeChar("((__half*)&"+var,")",string ("["));			 //var is the unmodified string
				ssResult <<  "["<<i<<"] = ((__half*)&"+ temp_var +"_masked_"+ to_string(currentIfLineNumber.first)+ ")["<<i<<"] ;\n";	
			}
			ssResult <<"}\n";
			if (!lhsMaskedInElseCond.empty()){
				ssResult <<"else{\n";
				for(string var : lhsMaskedInElseCond) {
				string temp_var = proccessArrayRefStr(var);
				ssResult << insertStringBeforeChar("((__half*)&"+var,")",string ("["));			
				ssResult <<  "["<<i<<"] = ((__half*)&"+ temp_var +"_masked_"+ to_string(currentIfLineNumber.first)+ "_else)["<<i<<"] ;\n";
				}		
				ssResult <<"}\n";	
			}
		}
		lhsMaskedInIffCond.clear();
		lhsMaskedInElseCond.clear();
		return ssResult.str();
		}
	string processCondStmtinIff(IfStmt* ifStatement){ //return processed cond statement for calculating mask e.g. if (x > 2) is the iffstmt at line no 32, return mask_id_32 = __hgt2(x, __float2half_rm(2.0));
		Expr* CondExpr = ifStatement->getCond();
		if (isa<BinaryOperator>(CondExpr))
		{
			BinaryOperator* binaryOp= dyn_cast<BinaryOperator>(CondExpr);
			Expr* LHS = binaryOp->getLHS();
			Expr* RHS = binaryOp->getRHS();
			SourceRange LHSRange = LHS->getSourceRange();
			LHSRange.setEnd(LHS->getLocEnd ().getLocWithOffset(1));
			SourceRange RHSRange = RHS->getSourceRange();
			RHSRange.setEnd(RHS->getLocEnd ().getLocWithOffset(1));
			string leftString = "";
			string rightString = "";
			LHS -> dump();
			RHS -> dump();
			
			if (IsHalf2Expr (LHS))
				leftString = getSourceTextFromSourceRange(&LHSRange);
			else{
				if (isa<IntegerLiteral>(LHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens())) {
					IntegerLiteral* intLiteral = dyn_cast<IntegerLiteral>(LHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens());	
					leftString = half2_mode?FLOAT2HALF2:FLOAT2HALF +to_string(intLiteral->getValue().getLimitedValue	()) + ")";
				} else if (isa<FloatingLiteral>(LHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens())){
						FloatingLiteral* floatLiteral = dyn_cast<FloatingLiteral>(LHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens());
						float fltEvalResult = 0.0;
						if (APFloat::semanticsSizeInBits(floatLiteral->getValue().getSemantics()) == 64) //detect double the ugly way
							fltEvalResult = (float) floatLiteral->getValue().convertToDouble () ;
						else
							fltEvalResult =  floatLiteral->getValue().convertToFloat();
					leftString = half2_mode?FLOAT2HALF2:FLOAT2HALF +to_string(fltEvalResult) + ")";			
				}else{
					leftString = "complex expr, to be developed";
					
				}
			}
			if (IsHalf2Expr (RHS))
				rightString = getSourceTextFromSourceRange(&RHSRange);
			else{
				if (isa<IntegerLiteral>(RHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens())) {
					IntegerLiteral* intLiteral = dyn_cast<IntegerLiteral>(RHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens());	
					rightString = half2_mode?FLOAT2HALF2:FLOAT2HALF +to_string(intLiteral->getValue().getLimitedValue	()) + ")";
				} else if (isa<FloatingLiteral>(RHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens())){
						FloatingLiteral* floatLiteral = dyn_cast<FloatingLiteral>(RHS->IgnoreImpCasts()->IgnoreCasts()->IgnoreParens());
						float fltEvalResult = 0.0;
						if (APFloat::semanticsSizeInBits(floatLiteral->getValue().getSemantics()) == 64) //detect double the ugly way
							fltEvalResult = (float) floatLiteral->getValue().convertToDouble () ;
						else
							fltEvalResult =  floatLiteral->getValue().convertToFloat();
					rightString = half2_mode?FLOAT2HALF2:FLOAT2HALF +to_string(fltEvalResult) + ")";			
				}else{
					rightString = "complex expr, to be developed";
					
				}
				
				
			}
			string result = "";
			if(DEBUG_MODE)
				cout<<"aaa  " <<leftString <<" " << rightString<<" \n";
				
			switch(binaryOp->getOpcode()){
				case BO_LT:
					result = half2_mode?HALF2_LT:HALF_GT + leftString + "," + rightString + ");\n"; 
				break;
				case BO_GT:
					result = half2_mode?HALF2_GT:HALF_GT + leftString + "," + rightString + ");\n"; 
				break;	
				case BO_EQ:
					result = half2_mode?HALF2_EQ:HALF_EQ + leftString + "," + rightString + ");\n"; 
				break;								
				case BO_NE:
					result = half2_mode?HALF2_NE:HALF_NE+ leftString + "," + rightString + ");\n"; 
				break;	
				case BO_LE:
					result = half2_mode?HALF2_LE:HALF_LE + leftString + "," + rightString + ");\n"; 
				break;
				case BO_GE:
					result = half2_mode?HALF2_GE:HALF_GE + leftString + "," + rightString + ");\n"; 
				break;						
				default:
				break;		 			

			}
			return half_type_string+" mask_if_"+ to_string(currentIfLineNumber.first) + " = " + result;
		} else return "";
	}
	string proccessArrayRefStr(string arrayRef){ //e.g. z_array[i] -> z_array_i
			string result = "" + arrayRef;
			int  foundBracket = result.find("[");
			if (foundBracket!= -1){
				result.replace(foundBracket,1,"_");
				removeCharsFromString(result,"["); //remove extra square brackets
				removeCharsFromString(result,"]");
			}
			return result;
		}
	string addMaskForExpr(Expr* expr, bool notFound = false){ //add suffix _masked to a var if it present in the lhsMaskedInIffCond list, set notFound= true to rewrite anyway (dont check if found or not)
			SourceRange * sourceRange = getSourceRangeIgnoreArrayRef(expr);
			sourceRange->setEnd (sourceRange->getEnd().getLocWithOffset(1));
			string lhsSource = "";
			SourceRange fullRange = expr->getSourceRange();
			fullRange.setEnd(fullRange.getEnd().getLocWithOffset(1));
			if(isa <DeclRefExpr>(expr->IgnoreImpCasts()->IgnoreParens ())){
				DeclRefExpr* declExpr = cast <DeclRefExpr> (expr->IgnoreImpCasts()->IgnoreParens ());
				lhsSource = declExpr->getDecl()->getNameAsString ();
			}else
				lhsSource = getSourceTextFromSourceRange(&fullRange);//need to include x[i] with x
			//~ string lhsSource = getSourceTextFromSourceRange(sourceRange);
			if(DEBUG_MODE){
				cout <<lhsSource<<" lhs source \n";
			//	lhsSource = proccessArrayRefStr(lhsSource);
			//	cout <<lhsSource<<" lhs source processed\n";
				for(int i=0; i<lhsMaskedInIffCond.size(); ++i)
					std::cout << lhsMaskedInIffCond[i] << ' ';
				cout <<"\n";
			}
			if (( find(lhsMaskedInIffCond.begin(), lhsMaskedInIffCond.end(), lhsSource) != lhsMaskedInIffCond.end() ) ||notFound) {
					//sthing strange here, need to reduce offset for correct string 
					//sourceRange->setEnd (sourceRange->getEnd().getLocWithOffset(-1));
					
					//~ TheRewriter.InsertTextBefore(sourceRange->getEnd().getLocWithOffset(1), "_masked");
					sourceRange->setEnd (sourceRange->getEnd().getLocWithOffset(-1));
					string lhsSourceProcessed = proccessArrayRefStr(lhsSource) + "_masked_" + to_string(currentIfLineNumber.first);
					if (insideElseBranch)
						lhsSourceProcessed += "_else";
					TheRewriter.ReplaceText(fullRange,lhsSourceProcessed);
				//	TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(sourceRange->getEnd()), "_masked_" + to_string(currentIfLineNumber.first)); //prevent duplicating with other iffs
					if(DEBUG_MODE)
					cout<<"added mask "<<lhsSource <<" \n";
			}
			//~ if (insideElseBranch)
				//~ return lhsSource+ "_else";
			return lhsSource;
		}

	void RewriteBinaryOp(SourceLocation leftLocation, SourceLocation rightLocation, BinaryOperator *E){
		//Context->getSourceManager().getExpansionLoc() is used in case macro ID is included
		//any case
		if(!insideIfCond && func_overload_mode)
			return;
		if (insideIfCond && (E->getOpcode()!= BO_Assign)){ 
				addMaskForExpr(E->getLHS());
				addMaskForExpr(E->getRHS());
			}
		switch(E->getOpcode()){
			case BO_AddAssign:
			case BO_Add:
				TheRewriter.InsertText(Context->getSourceManager().getExpansionLoc(leftLocation) ,half2_mode?HALF2_ADD:HALF_ADD);
				break;
			case BO_SubAssign:
			case BO_Sub:
				TheRewriter.InsertText(Context->getSourceManager().getExpansionLoc(leftLocation) ,half2_mode?HALF2_SUB:HALF_SUB);
				break;
			case BO_MulAssign:
			case BO_Mul:
				TheRewriter.InsertText(Context->getSourceManager().getExpansionLoc(leftLocation ),half2_mode?HALF2_MUL:HALF_MUL);
				break;
			case BO_DivAssign:
			case BO_Div	:
				TheRewriter.InsertText(Context->getSourceManager().getExpansionLoc(leftLocation) ,half2_mode?HALF2_DIV:HALF_DIV);
				break;
			case BO_Assign:
				if(DEBUG_MODE)
					cout<<"rewrite binary op Assign "<<insideIfCond<<"\n";
				if (insideIfCond){ //note down LHS to a vector, retrieve later
					//~ SourceRange * sourceRange = getSourceRangeIgnoreArrayRef(E->getLHS());
					//~ TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(sourceRange->getEnd()), "_masked");
					//~ string lhsSource = getSourceTextFromSourceRange(sourceRange);
					string lhsSource = addMaskForExpr(E->getLHS(),true);
					if(DEBUG_MODE)
						cout<<"added "<<lhsSource <<" to lhsMaskedInIffCond list";
					//~ lhsMaskedInIffCond.push_back(lhsSource);
					if (insideElseBranch)
						insertToVectorIfNotExist(lhsMaskedInElseCond, lhsSource);
					else
						insertToVectorIfNotExist(lhsMaskedInIffCond, lhsSource);
					//add the LHS sign to a list
					
				}
				return;//dont add , and ) at the end of this sw
				break;// never happens
			default:
				return; //dont modify the source code for other ops. // need case =
		}

		TheRewriter.ReplaceText(E->getOperatorLoc(), E->getOpcodeStr().size(), ",");
		TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(rightLocation), ")");
	}
	void RewriteLiterals(Expr* E){ //E can be floatingliteral, or any float-casted literals/function result
		//simple mode , treat all literals are float. Integer will be automaticly casted to float by nvcc
	//	E->dump();
		if(!insideIfCond && func_overload_mode)
			return;
			
		if (isa<CastExpr>(E)){
			CastExpr* castExpr = cast<CastExpr>(E);
			if(DEBUG_MODE){
				cout << "rewrite literal cast expr \n";
		//		if (!(isa<FloatingLiteral>(castExpr->getSubExpr()) || isa<IntegerLiteral>(castExpr->getSubExpr())) )
		//			return; 	//not literals, more complex funcall, or arrayref ... doing nothing.
				cout << "rewrite literal cast expr is cast literal \n";
			}
		}

		TheRewriter.InsertText(Context->getSourceManager().getExpansionLoc(E->getLocStart ()), half2_mode?FLOAT2HALF2:FLOAT2HALF);
		TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(E->getLocEnd ()), ")");
	}
	void RewriteUnaryOp(UnaryOperator * E){ //handle - (h2neg(x) call if x is not literal value)
		TheRewriter.ReplaceText(E->getOperatorLoc(), 1, " ");
		TheRewriter.InsertText(E->getOperatorLoc(),half2_mode?HALF2_NEG:HALF_NEG);
		TheRewriter.InsertTextAfterToken(E->getLocEnd(), ")");
	}
	void RewriteFunctionDecl(FunctionDecl* F){
			TheRewriter.ReplaceText(F->getReturnTypeSourceRange (), half_type_string);
		}
	void ProccessCompoundAssignOperator( CompoundAssignOperator *E){ //handle += *=  /= ..
		if(!insideIfCond && func_overload_mode)
			return;
		if (DEBUG_MODE)
			std::cout<<"visit  CompoundAssignOperator\n";
		if (CudaCode){
			string lhsTail = "";//some kind of expr DeclRefExpr needs this value to store the declname for lhs string we will insert to the left most location
			if (IsFloatingPointType(E->getComputationLHSType().getAsString()) && IsFloatingPointType(E->getComputationResultType().getAsString())){
				if (DEBUG_MODE)
					std::cout<<"visit  CompoundAssignOperator && cuda \n";
				CharSourceRange sourceRange =  CharSourceRange::getCharRange(E->getLHS()->getSourceRange());
				sourceRange.setEnd (sourceRange.getEnd().getLocWithOffset(1));			//somehow lhs is missing 1 last char (right most) ??? workaround here
				
				if (isa<UnaryOperator>(E->getLHS())){ //*d = *d +1.0 // deref unaryop
					if(DEBUG_MODE)
						std::cout<<"UnaryOperator lhs compound stmt \n";
					UnaryOperator* unaryOp = cast<UnaryOperator>(E->getLHS());
					CastExpr*  castExpr = nullptr;
					if (isa<CastExpr>(unaryOp->getSubExpr())){ //skip all CastExpr
							if(DEBUG_MODE)
								cout<<"cast sub expr \n";
							castExpr = cast<CastExpr>(unaryOp->getSubExpr());
							while (isa<CastExpr>(castExpr->getSubExpr())){//skip all CastExpr //preventive
									castExpr = cast<CastExpr>(castExpr->getSubExpr());
								}
						}
					sourceRange.setBegin (unaryOp->getOperatorLoc());
					sourceRange.setEnd (unaryOp->getSubExpr ()->getLocEnd());
					Expr * subExpr ;
					if (castExpr == nullptr )
						subExpr = unaryOp->getSubExpr ();
					else
						subExpr =castExpr->getSubExpr ();
					if (isa<DeclRefExpr>(subExpr)){ //insert declname to the right of the unaryoperator
							if(DEBUG_MODE)
								cout<<"cast DeclRefExpr \n";
							DeclRefExpr *declRefExpr = cast<DeclRefExpr>(subExpr);
							lhsTail= declRefExpr->getDecl ()->getNameAsString ();
						}
				}
				string lhsString = Lexer::getSourceText(sourceRange, Context->getSourceManager(), LangOptions(), 0) ;
				
				lhsString = lhsString + lhsTail ;
				if (insideIfCond){ //note down LHS to a vector, retrieve later
					SourceRange * sourceRange = getSourceRangeIgnoreArrayRef(E->getLHS());
					sourceRange->setEnd (sourceRange->getEnd().getLocWithOffset(1));	
					string lhsSource = getSourceTextFromSourceRange(sourceRange);	
					if(DEBUG_MODE)
						cout<<"added "<<lhsSource <<" to lhsMaskedInIffCond list";
					if (insideElseBranch)
						insertToVectorIfNotExist(lhsMaskedInElseCond, lhsSource);
					else
						insertToVectorIfNotExist(lhsMaskedInIffCond, lhsSource); //maybe we should consider check for duplicate data in lhsMaskedInIffCond. not necessary now
					//dirty trick here, if lhsString contains [, insert _masked before [. else, insert it at the end of lhsString
					int insertPosition = lhsString.find_first_of("[");;
					if(insertPosition== -1)
						insertPosition = lhsString.length();
					lhsString.insert(insertPosition,"_masked_"+ to_string(currentIfLineNumber.first));
					
				}		

				if (DEBUG_MODE)
					std::cout<<"lhsString "<< lhsString<<"\n";
		

				TheRewriter.InsertText(E->getLHS()->getExprLoc() ,lhsString + " = ");
				//~ TheRewriter.InsertTextBefore(E->getLHS()->getExprLoc().getLocWithOffset (-1) ,lhsString + " = ");
				SourceLocation rightLocation =	E->getLocEnd(); //3 = len " = "
				SourceLocation leftLocation  =	E->getLHS()->getExprLoc();
				RewriteBinaryOp(leftLocation,rightLocation,E);
				}
			}
		}

	bool VisitUnaryOperator (UnaryOperator * E){ //handle - (h2neg(x) call if x is not literal value)
		if(!insideIfCond && func_overload_mode)
			return true;
		if (CudaCode){
			if (DEBUG_MODE)
				cout<<"visit  UnaryOperator cuda\n";
			if(E->getOpcode() == UO_Deref)
				if(DEBUG_MODE)
					cout<<"UO deref found\n";
			if ( E->getOpcode() == UO_Minus  ){
				if (!isa<FloatingLiteral>(E->getSubExpr())){
					if(DEBUG_MODE)
						cout<<"neg unary op on var \n";

				Expr* childExpr = E->getSubExpr()->IgnoreCasts()->IgnoreImpCasts ();
				string varName = "";
				bool validHalf2Var = false;
				if (isa<ArraySubscriptExpr>(childExpr)){ //array ref

					ArraySubscriptExpr* subscriptExpr = cast<ArraySubscriptExpr>(childExpr);
					if (isa <ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts())) //2d arrays ?
						subscriptExpr = cast<ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts());

					if (isa <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts())){
						DeclRefExpr* declExpr = cast <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts());
						varName = declExpr-> getDecl ()-> getNameAsString();
						if (IsHalf2Var(varName) && CurrentFunc!=nullptr) validHalf2Var = true;
					}

					}
				else if (isa<DeclRefExpr>(childExpr)) { //single var name
					if(DEBUG_MODE)
						cout<<"DeclRefExpr \n";
					DeclRefExpr* declExpr = cast <DeclRefExpr> (childExpr);
					varName = declExpr-> getDecl ()-> getNameAsString();
					if (IsHalf2Var(varName) && CurrentFunc!=nullptr) validHalf2Var = true;
					}
				if (validHalf2Var)
					RewriteUnaryOp(E);
				else {//try to traverse back to its parent
								//this attemp failed for an unknown reason http://stackoverflow.com/questions/40871961/clang-astcontext-getparents-always-returns-an-empty-list
								// work around : try to get to this point from its parent (CastExpr)
					}
			}	else {
					if(DEBUG_MODE)
						cout<<"neg unary op on floating literal  \n";
					RewriteLiterals(E);
				}
		}
	}
		return true;
	}

//	void reWriteIndexLocation (Sour) //i => i/2 where suitable
	void processRewriteArraySubscriptIndex(Expr* E){
			if(DEBUG_MODE)
				cout <<" processRewriteArraySubscriptExpr \n";
			Expr* exprPlain = E->IgnoreImpCasts()->IgnoreParens ();
			if (isa <DeclRefExpr>(exprPlain)){ //simplest case a[tid]/ rewrite a[tid/2]
				//	cout<<"simplest case \n";
					DeclRefExpr* declRefExpr= cast<DeclRefExpr>(exprPlain);
					if(DEBUG_MODE)
						cout<<declRefExpr->getDecl ()->getNameAsString()<<"\n";
					
					if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
					{
				//		cout<<"inside iff \n";
						TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(declRefExpr->getLocEnd ()), "/2");
					}	
			}else if(isa <BinaryOperator>(exprPlain)){//recursively find tid loc //push back at LHS, popback at RHS
				//support 3 levels recursion/  i.e. ((a+b) + (c+d))*e
				vector<int> opcode_vec ; //BFS
				SourceLocation threadIdLocation ;
				bool foundIdx = false;
				BinaryOperator* binaryop = cast<BinaryOperator>(exprPlain);
				opcode_vec.push_back(binaryop->getOpcode());
				if(isa <DeclRefExpr>(binaryop->getLHS()->IgnoreImpCasts()->IgnoreParens ())) 
				{
					DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop->getLHS()->IgnoreImpCasts()->IgnoreParens ());
					if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
					{
						if(DEBUG_MODE)
							cout<<"caught LHS \n" ; 
						foundIdx = !foundIdx;
						threadIdLocation = declRefExpr->getLocEnd ();
					}
				} else if(isa <DeclRefExpr>(binaryop->getRHS()->IgnoreImpCasts()->IgnoreParens ())) 
					{
						DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop->getRHS()->IgnoreImpCasts()->IgnoreParens ());
						if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
						{
							if(DEBUG_MODE)
								cout<<"caught RHS \n" ; 
							foundIdx = !foundIdx;
							threadIdLocation = declRefExpr->getLocEnd ();
						}
					}
				else{ //binaryop
						if(isa <BinaryOperator>(binaryop->getLHS()->IgnoreImpCasts()->IgnoreParens ())){ //LHS
							BinaryOperator* binaryop1 = cast<BinaryOperator>((binaryop->getLHS()->IgnoreImpCasts()->IgnoreParens ()));
							if(!foundIdx)
								opcode_vec.push_back(binaryop1->getOpcode());
							if(isa <DeclRefExpr>(binaryop1->getLHS()->IgnoreImpCasts()->IgnoreParens ())) 
							{
								DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop1->getLHS()->IgnoreImpCasts()->IgnoreParens ());
								if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
								{
									if(DEBUG_MODE)
										cout<<"caught tid  LHS LHS\n" ; 
									foundIdx = !foundIdx;
									threadIdLocation = declRefExpr->getLocEnd ();
								}
							} else{
								//do nothing, need to refactor this code to recursive version to process this
								}
							if(isa <DeclRefExpr>(binaryop1->getRHS()->IgnoreImpCasts()->IgnoreParens ())) 
							{
								DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop1->getRHS()->IgnoreImpCasts()->IgnoreParens ());
								if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
								{
									if(DEBUG_MODE)
										cout<<"caught tid  LHS RHS\n" ; 
									foundIdx = !foundIdx;
									threadIdLocation = declRefExpr->getLocEnd ();
								}
							} else{
								//do nothing
								}
							if(!foundIdx)
								opcode_vec.pop_back()	;					
						}//end LHS
						if(isa <BinaryOperator>(binaryop->getRHS()->IgnoreImpCasts()->IgnoreParens ())){ //RHS
							BinaryOperator* binaryop1 = cast<BinaryOperator>((binaryop->getRHS()->IgnoreImpCasts()->IgnoreParens ()));
							if(!foundIdx)
								opcode_vec.push_back(binaryop1->getOpcode());
							if(isa <DeclRefExpr>(binaryop1->getLHS()->IgnoreImpCasts()->IgnoreParens ())) 
							{
								DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop1->getLHS()->IgnoreImpCasts()->IgnoreParens ());
								if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
								{
									if(DEBUG_MODE)
										cout<<"caught tid RHS LHS\n" ; 
									foundIdx = !foundIdx;
									threadIdLocation = declRefExpr->getLocEnd ();
								}
							} else{
								//do nothing, need to refactor this code to recursive version to process this
								}
							if(isa <DeclRefExpr>(binaryop1->getRHS()->IgnoreImpCasts()->IgnoreParens ())) 
							{
								DeclRefExpr* declRefExpr= cast<DeclRefExpr>(binaryop1->getRHS()->IgnoreImpCasts()->IgnoreParens ());
								if (IsThreadIdVar(declRefExpr->getDecl ()->getNameAsString()))
								{
									if(DEBUG_MODE)
										cout<<"caught tid RHS RHS\n" ; 
									foundIdx = !foundIdx;
									threadIdLocation = declRefExpr->getLocEnd ();
								}
							} else{
								//do nothing
								}
							if(!foundIdx)
								opcode_vec.pop_back()	;					
						}						
						
						
					
					}
			
			
			
			//process after found:
			if (foundIdx){
				bool valid_simpleFunc = true; //check if idx in form M[A*idx + B] where A must equal 1, which means all ops in opcode_vec != div or mul
				if(DEBUG_MODE)
					cout <<" end , found idx :"; 
				
				for (int i = 0;i<opcode_vec.size();i++){
					if(opcode_vec[i] == BO_Mul || opcode_vec[i] == BO_Div)
						valid_simpleFunc= false;		
					//cout<<opcode_vec[i]<< " ";
					}
				//cout <<"\n";
				
				
				if (valid_simpleFunc){
					if(DEBUG_MODE)
						cout <<"idX linear to array ref. rewrite \n";
					TheRewriter.InsertTextAfterToken(Context->getSourceManager().getExpansionLoc(threadIdLocation), "/2");
					
					}
				
			}
			else{
				cout <<"Not supported array subcript formula, not processed \n";
			}
			
		}
		}
	bool VisitArraySubscriptExpr(ArraySubscriptExpr* E){ //rewrite array access for half2 type
		if (!half2_mode)
			return true;
		Expr* base = E->getBase();
		Expr* index = E->getIdx();
		//base->IgnoreParens ()->();
		//index->IgnoreParens ()->dump();
		DeclRefExpr* declExpr = nullptr ;
		if (isa <DeclRefExpr> (base->IgnoreImpCasts()))
				declExpr = cast <DeclRefExpr> (E->getBase()->IgnoreImpCasts());

		if (declExpr == nullptr) return true;
		SourceRange *indexrange = new SourceRange(index->IgnoreImpCasts () ->getLocStart(), index ->IgnoreImpCasts ()->getLocEnd());
	//	SourceRange *baserange = new SourceRange(E->IgnoreImpCasts () ->getLocStart(), base->getLocEnd());
		
	//	baserange->setEnd (baserange->getEnd().getLocWithOffset(1));	
		if(DEBUG_MODE)
			cout <<"VisitArraySubscriptExpr : base " <<declExpr->getDecl ()->getNameAsString()<<"  index "<<getSourceTextFromSourceRange(indexrange)<<"\n";
		if (IsHalf2Var(declExpr->getDecl ()->getNameAsString()))
			processRewriteArraySubscriptIndex(index);
		
		return true;
		}

	bool VisitCallExpr(CallExpr *E){
		if (DEBUG_MODE){
			cout<<"visit call expr \n";
			string fullSrc = Lexer::getSourceText(CharSourceRange::getCharRange(E->getSourceRange()),Context->getSourceManager(),LangOptions(), 0);
			string toFirstArg = "none ";
			if(E->getNumArgs () >=1)
			 toFirstArg = Lexer::getSourceText(CharSourceRange::getCharRange(E->getLocStart(), E->getArg(0)->getLocStart().getLocWithOffset (-1) ),Context->getSourceManager(),LangOptions(), 0);

			cout << "fullSrc " << fullSrc << "\n";
			cout << "func name only " << toFirstArg << "\n";
		}
		if (E->getDirectCallee()!=nullptr && (E->getNumArgs () >=1)){

			FunctionDecl *directCallee = E->getDirectCallee();
			if (directCallee->hasAttr<CUDADeviceAttr>() ||directCallee->hasAttr<CUDAGlobalAttr>() ){
				RewriteFunctionCall(E);
				//rewrite args if they are floating literal or integer literals;
				for (int i = 0; i< E->getNumArgs ();i++){
					Expr* arg = E->getArg(i)->IgnoreImpCasts()->IgnoreImplicit()->IgnoreParens();
					if (isa<FloatingLiteral>(arg) || isa<IntegerLiteral>(arg))
						RewriteLiterals(arg);
				}
			}
		}
		return true;
	}

	bool VisitCastExpr(CastExpr *E){
		if (CudaCode){
			if (isa<UnaryOperator>(E->getSubExpr()))
			{
				UnaryOperator* unaryOp = cast<UnaryOperator>(E->getSubExpr());
				if (DEBUG_MODE)
					cout<<"VisitCastExpr cuda unaryop\n";
				if (unaryOp->getOpcode() == UO_Minus   && (isa<IntegerLiteral>(unaryOp->getSubExpr())))
				{	
					if(DEBUG_MODE)
						cout <<"rewrite literals  child of cast expr\n";
					RewriteLiterals(unaryOp); // rewrite literals because we will use float2half2_rn (neg intliteral);
				}
			}
		}
		return true;
		}
//TODO change IsFloatingPointType = IsSIMDType, provide short2
  bool VisitBinaryOperator(BinaryOperator *E){
	if (DEBUG_MODE){
		std::cout<<"visit binary operator \n";
		std::cout<<"binary operator inside iff cond, rewrite with masking\n";
		(E->getLHS())->dump();
		(E->getRHS())->dump();
	}
	QualType LHSType = E->getLHS()->getType();
	QualType RHSType = E->getRHS()->getType();
	if (IsFloatingPointType(LHSType.getAsString()) /*&& IsFloatingPointType(RHSType.getAsString())*/){
		if (CudaCode)	{
			if (DEBUG_MODE)
				std::cout<<"binary op in cuda detected \n";
			if(isa<CompoundAssignOperator>(E)){//handle += *=  /= ..
				CompoundAssignOperator * compoundAssignOp =  cast<CompoundAssignOperator>(E);
				ProccessCompoundAssignOperator(compoundAssignOp);
				return true;
				}
			else { //simple binary op
				SourceLocation leftLocation =  E->getLHS()->getExprLoc();
				SourceLocation rightLocation =	E->getRHS()->getLocEnd();
				if (isa<BinaryOperator>(E->getLHS())) {
					if (DEBUG_MODE)
						std::cout<<"LHS is Binary op\n";
					BinaryOperator *binaryOp = cast<BinaryOperator>(E->getLHS());
					while (isa<BinaryOperator>(binaryOp->getLHS())){ //recursively get to the left most position of binaryop
						binaryOp = cast<BinaryOperator>(binaryOp->getLHS());
					}
					leftLocation = binaryOp->getLHS()->getExprLoc();

				}else if (isa<BinaryOperator>(E->getRHS())) {
					if (DEBUG_MODE)
						std::cout<<"RHS is Binary op\n";
					BinaryOperator *binaryOp = cast<BinaryOperator>(E->getRHS());
					while (isa<BinaryOperator>(binaryOp->getRHS())){ //recursively get to the right most position of binaryop
						binaryOp = cast<BinaryOperator>(binaryOp->getRHS());
					}
					rightLocation = binaryOp->getRHS()->getLocEnd();
				}
				if(E->HasSideEffects(*Context) == false)
				{
					if(DEBUG_MODE)
						cout<< "no side effect \n";
					if (E->isEvaluatable (*Context)){//eval and return
						if(DEBUG_MODE)
							cout<< "isEvaluatable \n";
						APFloat evalResult(0.0);
						if (E->EvaluateAsFloat(evalResult,*Context)){
							float fltEvalResult = 0.0;
							if (APFloat::semanticsSizeInBits(evalResult.getSemantics()) == 64) //detect double the ugly way
								fltEvalResult = (float) evalResult.convertToDouble () ;
							else
								fltEvalResult =  evalResult.convertToFloat();
							if(DEBUG_MODE)
								cout<< "eval result " << fltEvalResult <<"\n";

							SourceRange  replaceRange;
							replaceRange.setBegin(E->getLHS()->getExprLoc());
							replaceRange.setEnd(E->getRHS()->getLocEnd());
							std::ostringstream ss;
							ss << fltEvalResult;
							std::string fltEvalString(ss.str());
							TheRewriter.ReplaceText(replaceRange, string(fltEvalString) );
							return true;
						}
					}
				}
				RewriteBinaryOp(leftLocation,rightLocation,E );
				bool LHSSubcript = false;
				bool RHSSubcript = false;
				Expr *LHSIgnoreParensCasts = E->getLHS()->IgnoreImpCasts()->IgnoreParens ();
				//refactor, funcall later. prob: uniaryops
				
				if(isa <ArraySubscriptExpr> (LHSIgnoreParensCasts ) ||  isa <DeclRefExpr>(LHSIgnoreParensCasts ) || isa <UnaryOperator>(LHSIgnoreParensCasts ) ){
					if(DEBUG_MODE)
						cout<<"LHS subscript \n";
					DeclRefExpr* declExpr = nullptr;
					if (isa <ArraySubscriptExpr> (LHSIgnoreParensCasts)){
						ArraySubscriptExpr* subscriptExpr = cast<ArraySubscriptExpr>(E->getLHS()->IgnoreImpCasts());
						if (isa <ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts()))
							subscriptExpr = cast<ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts());
						if (isa <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts()))
							declExpr = cast <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts());
					} else if (isa <UnaryOperator> (LHSIgnoreParensCasts)){ //only the case *var  when var is a pointer to a floating var

							UnaryOperator * unaryOp =  cast <UnaryOperator>(LHSIgnoreParensCasts);
							if(unaryOp->getOpcode() == UO_Deref)
								LHSIgnoreParensCasts = unaryOp->getSubExpr()->IgnoreCasts()->IgnoreImpCasts ();
						}
					if(isa <DeclRefExpr>(LHSIgnoreParensCasts))
						declExpr = cast <DeclRefExpr> (LHSIgnoreParensCasts);
					//subscriptExpr->dump();
					if (declExpr!=nullptr){
						string baseSrc = declExpr-> getDecl ()-> getNameAsString();
						if(DEBUG_MODE)
							cout<<"LHS subscript src " << baseSrc<<"\n";

							if (IsHalf2Var(baseSrc)&& CurrentFunc==nullptr) LHSSubcript = true;
							if (!IsHalf2Var(baseSrc)) LHSSubcript = true; //ofc
					}
				}
				Expr * RHSIgnoreParensCasts = E->getRHS()->IgnoreImpCasts()->IgnoreParens ();
				if(isa <ArraySubscriptExpr> (RHSIgnoreParensCasts) || isa <DeclRefExpr> (RHSIgnoreParensCasts) || isa <UnaryOperator>(RHSIgnoreParensCasts)){
					if(DEBUG_MODE)
						cout<<"RHS subscript \n";
					DeclRefExpr* declExpr = nullptr;
					if (isa <ArraySubscriptExpr> (RHSIgnoreParensCasts)){
						ArraySubscriptExpr* subscriptExpr = cast<ArraySubscriptExpr>(E->getRHS()->IgnoreImpCasts());
						if (isa <ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts()))
							subscriptExpr = cast<ArraySubscriptExpr>(subscriptExpr->getBase()->IgnoreImpCasts());
						if (isa <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts()))
							declExpr = cast <DeclRefExpr> (subscriptExpr->getBase()->IgnoreImpCasts());
					} else if (isa <UnaryOperator> (RHSIgnoreParensCasts)){ //only the case *var  when var is a pointer to a floating var

							UnaryOperator * unaryOp =  cast <UnaryOperator>(RHSIgnoreParensCasts);
							if(unaryOp->getOpcode() == UO_Deref)
								RHSIgnoreParensCasts = unaryOp->getSubExpr()->IgnoreCasts()->IgnoreImpCasts ();
						}
					if(isa <DeclRefExpr>(RHSIgnoreParensCasts))
						declExpr = cast <DeclRefExpr> (RHSIgnoreParensCasts);
					//subscriptExpr->dump();
					if (declExpr!=nullptr){
						string baseSrc = declExpr-> getDecl ()-> getNameAsString();
						if(DEBUG_MODE)
							cout<<"RHS subscript src " << baseSrc<<"\n";

							if (IsHalf2Var(baseSrc)&& CurrentFunc==nullptr) RHSSubcript = true;
							if (!IsHalf2Var(baseSrc)) RHSSubcript = true; //ofc
					}
				}


				if (isa<FloatingLiteral>(E->getLHS()->IgnoreImpCasts())|| LHSSubcript){
					if (DEBUG_MODE)
						std::cout<<"LHS is literal or casted \n";

					RewriteLiterals(E->getLHS());
				}
				if (isa<FloatingLiteral>(E->getRHS()->IgnoreImpCasts()) || RHSSubcript){
					if (DEBUG_MODE)
						std::cout<<"RHS is literal or casted \n";
					RewriteLiterals(E->getRHS());
				}

		}
	}
	}
	return true;
  }

  bool VisitParmVarDecl(ParmVarDecl *Decl)
  {
	  if(DEBUG_MODE){
	  cout<<"visit ParamDecl \n";
	  cout<<(Decl->getOriginalType()).getAsString()<<"\n";
	  cout<<Decl->getNameAsString()<<"\n";
	}	
	  string typeStr = (Decl->getOriginalType()).getAsString(); //currently match str with double/float. don't consider whether it's pointer or sth
	  //Decl->dump();
	  if(CudaCode)
	  {
		  if (IsHalf2Var(Decl->getNameAsString()))
		  {
			  //TODO: add declName to a list to retrieve later.
			  int dimension = count(typeStr.begin(), typeStr.end(), '*'); //dirty hack for pointer type. need to read Clang Docs later
			  //replace double * with __half2 *
			  TheRewriter.ReplaceText(Decl->getTypeSpecStartLoc(), (Decl->getOriginalType()).getAsString().size(), half_type_string+" " + string(dimension, '*'));

				//~ if (CurrentFunc!= nullptr)
					//~ half2Vars.push_back( make_pair(Decl->getNameAsString(),CurrentFunc->getNameAsString()));
				//~ else
					//~ half2Vars.push_back( make_pair(Decl->getNameAsString(),""));

		  } else if (IsIntegerType(typeStr)){
			  if (CurrentFunc!=nullptr)
				NoThreadVars.push_back( make_pair(Decl->getNameAsString(), CurrentFunc->getNameAsString()));
				else //never happens ??
				NoThreadVars.push_back( make_pair(Decl->getNameAsString(), ""));
		  }
	 }

	  return true;
  }
	bool VisitVarDecl(VarDecl* Decl){
		if(DEBUG_MODE){
		  cout<<"visit VarDecl \n";
		  cout<<(Decl->getType()).getAsString()<<"\n";
		  cout<<Decl->getNameAsString()<<"\n";
		  cout<<"has init "<< Decl->hasInit()<<"\n";
	  }
	  string typeStr = (Decl->getType().getUnqualifiedType()).getAsString();
	  //~ cout<<"is first " <<Decl-> isFirstDecl ()<<" \n";
	  if(DEBUG_MODE){
		cout <<"has cont attr "<<Decl->hasAttr<CUDAConstantAttr>()<<"\n";
		Decl->dump();
	}
	  if (!(Decl->isLocalVarDecl () ||Decl->hasAttr<CUDAConstantAttr>() || Decl->hasAttr<CUDADeviceAttr>() )) return true;//visit parm decl was proccessed.
	  if(CudaCode || Decl->hasAttr<CUDAConstantAttr>() || Decl->hasAttr<CUDADeviceAttr>() )
	  {
		  if(DEBUG_MODE)
			cout<<" vardecl in cuda\n";
		  if (IsHalf2Var(Decl->getNameAsString())){
				if(DEBUG_MODE)
					cout<<" vardecl is half2 type\n";
			 ; //handle half2 decl / did in dectect_half2_vars.cpp

			// if(Decl->hasAttr<CUDAConstantAttr>() || Decl->hasAttr<CUDADeviceAttr>()) return true; //currently dont attemp to rewrite global decl ?/ will do it later
			 //rewrite float into __half2
			  int dimension = count(typeStr.begin(), typeStr.end(), '*'); //dirty hack for pointer type. need to read Clang Docs later
			  SourceLocation startLoc = Decl->getTypeSpecStartLoc();
			  if (std::find(rewrittenLocation.begin(), rewrittenLocation.end(), startLoc) == rewrittenLocation.end()){ //not found, push_back + rewrite

				rewrittenLocation.push_back(startLoc);

				TheRewriter.ReplaceText(startLoc, (Decl->getType().getUnqualifiedType()).getAsString().size(), half_type_string + string(dimension, '*'));
				//have problem with group decl, theRewriter will replace float multiple times. Workaround version : store rewritten sourcelocation into a vector. check if it existed ?
			}
			if ( Decl->hasInit()){ //rewrite init floating literal
				Expr* InitStmt = Decl->getInit();
				if (isa<FloatingLiteral>(InitStmt) || isa<IntegerLiteral>(InitStmt)){
					if(DEBUG_MODE)
						cout<<"init literal \n";
					RewriteLiterals(InitStmt);

					}

				}

		  } else if (IsIntegerType(typeStr)){//detect variables used for thread index
			 if (Decl->hasInit()){
				 Expr* InitStmt = Decl->getInit();
				 string InitSrc = Lexer::getSourceText(CharSourceRange::getCharRange(InitStmt->getSourceRange()),Context->getSourceManager(),LangOptions(), 0);
				 if(DEBUG_MODE)
					cout<<"init "<<InitSrc<<"\n";
				if (IsThreadIDRelated(InitSrc)){
					if (CurrentFunc!=nullptr)
						IndexVars.push_back( make_pair(Decl->getNameAsString(),CurrentFunc->getNameAsString()));
					else //never happens ??
						IndexVars.push_back( make_pair(Decl->getNameAsString(),""));
					}
			 }
		  }

	  }
	return true;
	}

	bool VisitIfStmt(IfStmt* IfStatement){
	if(!half2_mode) return true; //half mode, doesn't have diverging prob, do nothing
		if(DEBUG_MODE)
			cout <<"visit iff  decl " << CudaCode <<"\n";
		Expr* CondExpr = IfStatement->getCond();
		if (isa<BinaryOperator>(CondExpr))
		{
			BinaryOperator* BinaryOp= dyn_cast<BinaryOperator>(CondExpr);
			if(DEBUG_MODE){
				cout<<"dump condExpr"<<"\n";
				CondExpr->dump();
			}
			if(IsHalf2Expr(BinaryOp->getLHS())|| IsHalf2Expr(BinaryOp->getRHS())){
				if(DEBUG_MODE){
					cout<<"SIMD cond! diverging " <<"\n";
					cout<<"begin line "<<Context->getSourceManager().getSpellingLineNumber(IfStatement->getLocStart())<<"\n";
					cout<<"end line "<<Context->getSourceManager().getSpellingLineNumber(IfStatement->getLocEnd())<<"\n";
				}
				currentIfLineNumber = make_pair(Context->getSourceManager().getSpellingLineNumber(IfStatement->getLocStart()),Context->getSourceManager().getSpellingLineNumber(IfStatement->getLocEnd()) );
				currentIfStmt = IfStatement;
				Stmt * elseStmt = IfStatement->getElse ();
				if(elseStmt != nullptr){
					currentElseLineNumber = make_pair(Context->getSourceManager().getSpellingLineNumber(elseStmt->getLocStart()),Context->getSourceManager().getSpellingLineNumber(elseStmt->getLocEnd()) );
				if(DEBUG_MODE)
					cout <<"else begin  "<<currentElseLineNumber.first<<"  "<<currentElseLineNumber.second <<"\n";

				}
			}
		//	string LHSOp = Lexer::getSourceText(CharSourceRange::getCharRange(BinaryOp->getLHS()->getSourceRange()),Context->getSourceManager(),LangOptions(), 0);
		//	string RHSOp= Lexer::getSourceText(CharSourceRange::getCharRange(BinaryOp->getRHS()->getSourceRange()),Context->getSourceManager(),LangOptions(), 0);
		//	cout<<"binary ops iff cond " << LHSOp << " xxx " << RHSOp <<"\n";
			
		}

		
		
		return true;	
}


  bool VisitStmt(Stmt *s) { //too general, consider special case first
    // Only care about If statements.
    if(!half2_mode) return true; //below is for divering & conditional stmts rewriting. No need in half_mode
    int current_line = Context->getSourceManager().getSpellingLineNumber(s->getLocStart());
/*	if(first_statement ){ // a little trick here to append #include list
		SourceLocation ST = s->getSourceRange().getBegin();
		cout <<"first statement \n";
		TheRewriter.InsertText(ST, "test include", true, true);			
		first_statement = false;
	}*/
	if(DEBUG_MODE)
		cout<<" stmt begin line "<<current_line<<"\n";
	//before updating last stmt line. search through ending lines of if/else stmt to update insideIfCond
	if (current_line<currentIfLineNumber.second && insideIfCond==false){
		insideIfCond = true;
		}
	if (current_line>=currentElseLineNumber.first && current_line<=currentElseLineNumber.second && insideIfCond==true){
		insideElseBranch = true;
		}
	//apply to multiple files ?
	if (current_line>currentIfLineNumber.second && insideIfCond==true){
		insideIfCond = false;
		insideElseBranch = false;
		//post process, apply masked, insert masked to the location right before if ;
		stringstream SSBefore;
		SSBefore << "// Begin iff cond  ,, doing something here" << "\n" 
		
		<<"//insert ifmask calculation here"<<"\n" 
		<< processCondStmtinIff(currentIfStmt)
		<< processAssignPreIff(currentIfStmt);
		SourceLocation ST = currentIfStmt->getSourceRange().getBegin();
		TheRewriter.InsertText(ST, SSBefore.str(), true, true);	
		//comment out "if (...)". leave the brackets untouched{}	
		TheRewriter.InsertText(currentIfStmt->getIfLoc (), "/*", true, true);	
		TheRewriter.InsertText(currentIfStmt->getThen ()->getLocStart (), "*/", true, true);	
		Stmt * elseStmt = currentIfStmt->getElse ();
		if(elseStmt != nullptr){
			TheRewriter.InsertText(currentIfStmt->getElseLoc (), "/*", true, true);	
			TheRewriter.InsertText(elseStmt->getLocStart (), "*/", true, true);	
			
			}
		

		std::stringstream SSAfter;
		SSAfter << "\n// End iff cond, add predicate iff here \n"
		<<processAssignPostIff(currentIfStmt);
		ST = currentIfStmt->getSourceRange().getEnd().getLocWithOffset(1);
		TheRewriter.InsertText(ST, SSAfter.str(), true, true);
		
		
		
		}
//	lastStmtLine = Context->getSourceManager().getSpellingLineNumber(s->getLocStart());
	
      //  s->dump();
        //~ cout <<"Trying to get parents \n";
        //~ const Stmt& currentStmt = *s;
        //~ const auto& parents  = Context->getParents(currentStmt);
        //~ auto it = Context->getParents(currentStmt).begin();
        //~ if(it == Context->getParents(currentStmt).end())
            //~ cout<< "parents not found\n";
        //~ cout<<"parents size "<< parents.size() <<": \n";
        //~ if (!parents.empty()){
            //~ for (int i = 0; i< parents.size(); i++ ){
                //~ cout<<"parent at "<< i <<": \n";
                //~ const Stmt* parentStmt =  parents[i].get<Stmt>();
        //~ //        parentStmt->dump();
            //~ }

        //~ }



    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *f) {
    // Only function definitions (with bodies), not declarations. //will care header files later
    CurrentFunc = f;
	if ((f->hasAttr<CUDAGlobalAttr>() || f->hasAttr<CUDADeviceAttr>()) && isProcessingFunction(f->getNameInfo().getName().getAsString())){
		CudaCode = true;
		//~ cout << f->getNameInfo().getName().getAsString() <<"\n";
		if (IsHalf2Var(f->getNameInfo().getName().getAsString())){ //funcName funcName is determined by the half2_var_dectect phase
			// rewrite return type of f
			//~ cout<<"rewrite return type of F\n";
			RewriteFunctionDecl(f);
		}
	}
	else
		CudaCode = false;

    if (f->hasBody()) {

      Stmt *FuncBody = f->getBody();

      // Type name as string
      QualType QT = f->getReturnType();
      std::string TypeStr = QT.getAsString();

      // Function name
      DeclarationName DeclName = f->getNameInfo().getName();
      std::string FuncName = DeclName.getAsString();

      // Add comment before
      std::stringstream SSBefore;
      SSBefore << "// Begin function " << FuncName << " returning " << TypeStr
               << "\n";
      
      std::cout <<"visit FucntionDecl "<< FuncName<<"\n";
      SourceLocation ST = f->getSourceRange().getBegin();
      TheRewriter.InsertText(ST, SSBefore.str(), true, true);

      // And after
      std::stringstream SSAfter;
      SSAfter << "\n// End function " << FuncName;
      	//~ if (f->hasAttr<CUDAGlobalAttr>() || f->hasAttr<CUDADeviceAttr>())
		//~ {
			//~ std::cout <<"inside cuda mod  func decl\n";
			 //~ SSAfter << "\n// End cuda function " << FuncName ;
		//~ }
      ST = FuncBody->getLocEnd().getLocWithOffset(1);
      TheRewriter.InsertText(ST, SSAfter.str(), true, true);
    }

		      DeclarationName DeclName = f->getNameInfo().getName();
      std::string FuncName = DeclName.getAsString();
		std::cout <<" end visit FucntionDecl "<< FuncName <<"\n";
    return true;
  }

  bool VisitCUDADeviceAttr(CUDADeviceAttr* Attr){
	  if (CurrentFunc!= nullptr)
		if(DEBUG_MODE)
			cout <<"visit cuda Dev attr end of func "<<CurrentFunc->getNameAsString () <<" \n";
	  CurrentFunc = nullptr;
	  //~ CudaCode = false;
	  return true;
	  }

	bool VisitCUDAGlobalAttr(CUDAGlobalAttr* Attr){
		if (CurrentFunc!= nullptr)
			if(DEBUG_MODE)
				cout <<"visit cuda Global Attr end of func "<<CurrentFunc->getNameAsString () <<"\n";
		CurrentFunc = nullptr;
		//~ CudaCode = false;
		return true;
		}
private:
  Rewriter &TheRewriter;
  bool CudaCode; //detect whether we are parsing cuda code or host code
  ASTContext *Context;
  FunctionDecl* CurrentFunc; //not a good way to get current func, will change later
  IfStmt* currentIfStmt;
	vector<SourceLocation > rewrittenLocation;
	int lastStmtLine;
	int lastCompoundAssignLine;//compound assign, we append leff hand side , need to track it
	int lastCompountAssignOffset;//remember last compound assign offset, append it to the last position of the line 
	bool insideIfCond;
	bool insideElseBranch;
	pair<int,int> currentIfLineNumber;
	pair<int,int> currentElseLineNumber;
	vector<string > lhsMaskedInIffCond; //store masked values in LHS, rewrite any intances of this list in RHS; to be removed after exit iff
	vector<string > lhsMaskedInElseCond; //store masked values in LHS, rewrite any intances of this list in RHS; to be removed after exit iff	
};


// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.

class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R, ASTContext *Context) : Visitor(R,Context) {}


	virtual void HandleTranslationUnit(ASTContext &Context) {

		Visitor.TraverseDecl(Context.getTranslationUnitDecl());
	}


private:
  MyASTVisitor Visitor;

};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    SourceManager &SM = TheRewriter.getSourceMgr();
    llvm::errs() << "** EndSourceFileAction for: "
                 << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";

    // Now emit the rewritten buffer.
  //  TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());
    std::error_code error_code;
	llvm::raw_fd_ostream outFile("output.cu", error_code, llvm::sys::fs::F_None);
	if(func_overload_mode){
		if(half2_mode)
			outFile<<"#include <half2_operator_overload.cuh>\n";
		else
			outFile<<"#include <half_operator_overload.cuh>\n";
		}

    TheRewriter.getEditBuffer(SM.getMainFileID()).write(outFile);
    outFile.close();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {

    llvm::errs() << "** Creating AST consumer for: " << file << "\n";
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter,&CI.getASTContext());
  }

private:
  Rewriter TheRewriter;
};
void ReadHalf2VarsList(){
	ifstream myFile("half2VarsList.txt");
	string varName,funcName;


	while(myFile >> varName >> funcName){
		if (funcName.compare("null") == 0){
			funcName = "";
			half2Vars.push_back( make_pair(varName,funcName));
		} else{
			half2Vars.push_back( make_pair(varName,funcName));
			half2Vars.push_back( make_pair(funcName,funcName)); //note, funcname funcname to quickly search for favorable functions 
		}
	//	std::cout << varName << ' ' << funcName << std::endl;
	}


	}

vector<pair<string,string>> parse_config(){
	vector<pair<string,string> > result;
	std::ifstream infile("rewrite.conf");
	if(!infile){
			cout<<"could not read configuration file (rewrite.conf), use default setting (simd_mode=true, function_overload=true)";
			return result;
		}
	std::string line;
	while( std::getline(infile, line) )
	{
		std::istringstream infile(line);
		std::string key;
		if( std::getline(infile, key, '=') )
		{
			std::string value;
			if( std::getline(infile, value) ) 
				result.push_back(make_pair(key, value));
		}
	}
	return result;
}	

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, ToolingSampleCategory);	
  //CommonOptionsParser op(argc, argv);
  ReadHalf2VarsList();
	vector<pair<string,string>> options  = parse_config();
	if (options.size()!=0){
			for (int i =0; i<options.size();i++){
					cout<<options[i].first<<" "<<options[i].second<<"\n";
						if (options[i].first=="simd_mode"){
							cout<<" detected conf simd_mode \n";
							if (options[i].second=="true"){
								half2_mode = true;
								half_type_string = "__half2";
							}
							else{
								half2_mode = false;
								half_type_string = "__half";
							}
						}

						if (options[i].first=="operator_overload"){
							cout<<" detected conf operator_overload \n";
							if (options[i].second=="true" )
								func_overload_mode = true;
							else
								func_overload_mode = false;
							}
							
				}
		}

  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  // ClangTool::run accepts a FrontendActionFactory, which is then used to
  // create new objects implementing the FrontendAction interface. Here we use
  // the helper newFrontendActionFactory to create a default factory that will
  // return a new MyFrontendAction object every time.
  // To further customize this, we could create our own factory class.
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
